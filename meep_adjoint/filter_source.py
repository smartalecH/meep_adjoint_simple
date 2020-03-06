import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, PchipInterpolator
from scipy.special import erf


class FilteredSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,dt,T,time_src,min_err=1e-6):
        dt = dt/2
        self.dt = dt
        self.center_frequency=center_frequency
        self.frequencies=frequencies
        self.time_src=time_src
        self.min_err = min_err
        f = self.func()

        # calculate dtft of input signal
        signal_t = np.array([time_src.swigobj.current(t,dt) for t in np.arange(0,T,dt)]) # time domain signal
        signal_dtft = np.exp(1j*2*np.pi*frequencies[:,np.newaxis]*np.arange(0,signal_t.size)[np.newaxis,:]*dt)@signal_t # vectorize dtft for speed

        # find the frequency linear phase shift needed to time shift the impulse response in time domain
        self.time_shift = 0
        self.time_shift_n = np.round(self.time_shift/dt)
        
        phi = np.exp(1j*2*np.pi*frequencies*self.time_shift/dt)         

        # multiply sampled dft of input signal with filter transfer function
        H = signal_dtft * frequency_response #* phi

        self.estimate_impulse_response(H)

        y = [self(t) for t in np.arange(-T,T,dt)]

        plt.figure()
        plt.plot(np.arange(-T,T,dt),np.abs(y))

        y_hat = signal_dtft = np.exp(1j*2*np.pi*frequencies[:,np.newaxis]*np.arange(0,len(y))[np.newaxis,:]*dt)@y

        '''plt.figure()
        plt.plot(frequencies,np.unwrap(np.angle(phi)),label='phi')    
        plt.plot(frequencies,np.unwrap(np.angle(signal_dtft * frequency_response)),label='before')
        plt.plot(frequencies,np.unwrap(np.angle(H)),label='after')
        plt.plot(frequencies,np.unwrap(np.angle(y_hat)),label='final')
        plt.legend()'''

        plt.show()
        quit()

        # initialize super
        super(FilteredSource, self).__init__(src_func=f,center_frequency=self.center_frequency,is_integrated=False)
        
    def gaussian(self,f,f0,fwidth):
        return np.exp(-0.5*((f-f0)/fwidth)**2) * np.exp(1j*2*np.pi*f*self.time_shift/self.dt) 
    def antiderivative(self,f,n,f0,fwidth,T):
        m = self.time_shift/2
        a = np.sqrt(np.pi/2)*fwidth
        phase = np.exp(-2*np.pi*n*T*(np.pi*n*T*fwidth*fwidth + 1j*f0))
        #phase = np.exp(-2*np.pi*(n*T+m)*(np.pi*(n*T+m)*fwidth*fwidth + 1j*f0))
        kernel = erf(f/(np.sqrt(2)*fwidth)-f0/(np.sqrt(2)*fwidth)+1j*np.sqrt(2)*np.pi*n*fwidth*T)
        #kernel = erf(f/(np.sqrt(2)*fwidth)-f0/(np.sqrt(2)*fwidth)-1j*np.sqrt(2)*np.pi*fwidth*(m+n*T))
        return a*phase*kernel
    def dtft_gaussian(self,n,f0,fwidth,T):
        f_start = 0
        f_end = 1/T
        return T*(self.antiderivative(f_end,n,f0,fwidth,T) - self.antiderivative(f_start,n,f0,fwidth,T))
    def __call__(self,t):
        n = int(np.round((t)/self.dt))
        #print(t/self.dt,n)
        vec = self.dtft_gaussian(n,self.frequencies,self.gaus_widths,self.dt)
        # simple RBF with gaussian kernel reduces to inner product at time step
        return np.dot(vec,self.nodes)
    
    def func(self):
        def _f(t): 
            return self(t)
        return _f
    
    def estimate_impulse_response(self,H):
        '''
        find gaussian weighting coefficients.

        TODO use optimizer to find optimal gaussian widths
        '''
        # Use vandermonde matrix to calculate weights of each gaussian.
        # Each gaussian is centered at each frequency point
        def rbf_l2(fwidth):
            vandermonde = self.gaussian(self.frequencies[:,np.newaxis],self.frequencies[np.newaxis,:],fwidth)
            nodes = np.matmul(np.linalg.pinv(vandermonde),H)
            H_hat = np.matmul(vandermonde,nodes)
            l2_err = np.sum(np.abs(H-H_hat)**2)
            return nodes, l2_err
        
        df = self.frequencies[2] - self.frequencies[1]
        err_high = True
        fwidth = 1/self.time_src.width
        
        # Iterate through smaller and smaller widths until error is small enough or width is distance between frequency points
        if df < fwidth:
            while err_high:
                nodes, l2_err = rbf_l2(fwidth)
                if l2_err < self.min_err or fwidth < df:
                    err_high = False
                else:
                    fwidth = 0.8 * fwidth
                print(l2_err)
        self.gaus_widths = fwidth
        self.nodes = nodes

        from matplotlib import pyplot as plt

        temp = self.gaussian(self.frequencies[:,np.newaxis],self.frequencies,fwidth)
        i_hat = np.inner(self.nodes,temp)

        plt.figure()
        plt.subplot(2,1,1)
        plt.semilogy(self.frequencies,np.abs(H)**2)
        plt.semilogy(self.frequencies,np.abs(i_hat)**2)

        plt.subplot(2,1,2)
        plt.plot(self.frequencies,np.unwrap(np.angle(H)))
        plt.plot(self.frequencies,np.unwrap(np.angle(i_hat)))
        '''plt.show()
        quit()'''