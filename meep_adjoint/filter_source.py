import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, PchipInterpolator


class FilteredSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,dt,T,time_src,min_err=1e-6):
        self.center_frequency=center_frequency
        self.frequencies=frequencies
        self.time_src=time_src
        self.min_err = min_err
        f = self.func()

        # calculate dtft of input signal
        signal_t = np.array([time_src.swigobj.current(t,dt) for t in np.arange(0,T,dt)])
        signal_dtft = np.zeros((frequencies.shape),dtype=np.complex128)
        for n, st in enumerate(signal_t):
            for fi, f in enumerate(frequencies):
                signal_dtft[fi] += np.exp(1j*2*np.pi*f*n*dt)*st

        # multiply sampled dft of input signal with filter transfer function
        H = signal_dtft #* frequency_response

        # fit final frequency response to rbf
        H_rbf = Rbf(frequencies, H, function='gaussian')
        f_rbf = np.arange(0,1/dt,1/T)

        # estimate impulse response of final frequency response using ifft
        h = np.flipud(np.fft.ifft(H_rbf(f_rbf))) # flip signal becuase fft convention is backwards of meeps
        t_h = np.arange(0,T,dt)

        # fit impulse response to function to make implementation easy
        h_rbf = PchipInterpolator(t_h, h)  # we don't need a ton of extra accuracy at this point -- just speed
        #signal_fourier_transform = np.array([time_src.fourier_transform(f) for f in self.frequencies])
        # view results
        '''plt.figure()
        plt.subplot(2,1,1)
        plt.semilogy(t_h,np.abs(h))
        #plt.plot(t_h,np.abs(h_rbf(t_h)),'--')
        #plt.xlim(1/1.7,1/1.4)

        plt.subplot(2,1,2)
        plt.plot(t_h,np.unwrap(np.angle(h)))
        #plt.plot(t_h,np.unwrap(np.angle(h_rbf(t_h))),'--')
        #plt.xlim(1/1.7,1/1.4)
        plt.show()
        quit()'''

        def f_temp(t):
            return np.asscalar(h_rbf(t))

        #self.estimate_impulse_response()
        # initialize super
        super(FilteredSource, self).__init__(src_func=f_temp,center_frequency=self.center_frequency,is_integrated=False)
        
    def gaussian(self,f,f0,fwidth):
        return np.exp(-(f-f0)**2/(2*fwidth**2))

    def __call__(self,t):
        # simple RBF with gaussian kernel reduces to inner product at time step
        return np.dot(self.gauss_t(t,self.frequencies,self.gaus_widths),self.nodes)
    
    def func(self):
        def _f(t): 
            return self(t)
        return _f
    
    def estimate_impulse_response(self):
        '''
        find gaussian weighting coefficients.

        TODO use optimizer to find optimal gaussian widths
        '''
        # Use vandermonde matrix to calculate weights of each gaussian.
        # Each gaussian is centered at each frequency point
        h = self.frequency_response
        def rbf_l2(fwidth):
            vandermonde = np.zeros((self.frequencies.size,self.frequencies.size),dtype=np.complex128)
            for ri, rf in enumerate(self.frequencies):
                for ci, cf in enumerate(self.frequencies):
                    vandermonde[ri,ci] = self.gauss_f(rf,cf,fwidth)
            
            nodes = np.matmul(np.linalg.pinv(vandermonde),h)
            h_hat = np.matmul(vandermonde,nodes)
            l2_err = np.sum(np.abs(h-h_hat)**2)
            return nodes, l2_err
        
        df = self.frequencies[2] - self.frequencies[1]
        err_high = True
        fwidth = 1/self.time_src.width
        
        # Iterate through smaller and smaller widths until error is small enough or width is distance between frequency points
        while err_high:
            nodes, l2_err = rbf_l2(fwidth)
            if l2_err < self.min_err or fwidth < df:
                err_high = False
            else:
                fwidth = 0.5 * fwidth
        self.gaus_widths = fwidth
        self.nodes = nodes

        from matplotlib import pyplot as plt

        temp = self.gauss_f(self.frequencies[:,np.newaxis],self.frequencies,fwidth)
        i_hat = np.inner(self.nodes,temp)

        plt.figure()
        plt.subplot(2,1,1)
        plt.semilogy(self.frequencies,np.abs(h)**2)
        plt.semilogy(self.frequencies,np.abs(i_hat)**2)

        plt.subplot(2,1,2)
        plt.plot(self.frequencies,np.unwrap(np.angle(self.frequency_response)))
        plt.plot(self.frequencies,np.unwrap(np.angle(i_hat)))
    
    def gauss_t(self,t,f0,fwidth):
        s = 5
        w = 1.0 / fwidth
        t0 = w * s
        tt = (t - t0)
        amp = 1.0 / (-1j*2*np.pi*f0)
        return amp * np.exp(-tt * tt / (2 * w * w))*np.exp(-1j*2 * np.pi * f0 * tt) #/ np.sqrt(2*np.pi)
    
    def gauss_f(self,f,f0,fwidth):
        s = 5
        w = 1.0 / fwidth
        t0 = w * s
        omega = 2.0 * np.pi * f
        omega0 = 2.0 * np.pi * f0
        delta = (omega - omega0) * w
        amp = self.center_frequency/f0#/np.sqrt(omega0)#1j / (omega0)
        return amp * w * np.exp(1j*omega*t0) * np.exp(-0.5 * delta * delta)