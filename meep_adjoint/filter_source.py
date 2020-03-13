import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, PchipInterpolator
from scipy.special import erf

class FilteredSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,dt,T,time_src=None):
        dt = dt/2 # divide by two to compensate for staggered E,H time interval
        self.dt = dt
        self.frequencies=frequencies
        self.center_frequencies=frequencies
        df = frequencies[1]-frequencies[0]
        min_dtft_time = 1/df
        self.N = np.round(T/dt)
        f = self.func()

        if time_src:
            # calculate dtft of input signal
            signal_t = np.array([time_src.swigobj.current(t,dt) for t in np.arange(0,T,dt)]) # time domain signal
            signal_dtft = np.exp(1j*2*np.pi*frequencies[:,np.newaxis]*np.arange(0,signal_t.size)[np.newaxis,:]*dt)@signal_t # vectorize dtft for speed        
        else:
            signal_dtft = 1
        
        # multiply sampled dft of input signal with filter transfer function
        H = signal_dtft * frequency_response

        # estimate the impulse response using a sinc function RBN
        self.nodes, self.err = self.estimate_impulse_response(H)

        # Check the corresponding dtft response
        t = np.arange(0,T,dt)
        x = [self(t1) for t1 in t]
        X = np.exp(1j*2*np.pi*frequencies[:,np.newaxis]*t[np.newaxis,:])@x
        final_err = np.sum(np.abs(X-H)**2 / np.abs(H)**2)
        print("final relative error: ",final_err)

        #plt.figure()
        plt.subplot(3,1,2)
        plt.semilogy(t,np.abs(x))
        plt.xlabel('Time')
        
        #plt.figure()
        plt.subplot(3,2,5)
        plt.semilogy(frequencies,np.abs(H))
        plt.semilogy(frequencies,np.abs(X),'--')
        plt.xlabel('Freq')
        plt.subplot(3,2,6)
        plt.plot(frequencies,np.abs(H))
        plt.plot(frequencies,np.abs(X),'--')
        plt.xlabel('Freq')
        
        plt.tight_layout()
        #plt.savefig('numfreqs_{}.png'.format(frequencies.size))
        #plt.show()
        #quit()
        

        # initialize super
        super(FilteredSource, self).__init__(src_func=f,center_frequency=center_frequency,is_integrated=False)

    def sinc(self,f,f0):
        omega = 2*np.pi*f
        omega0 = 2*np.pi*f0
        num = np.where(omega == omega0, self.N, (1-np.exp(1j*(self.N+1)*(omega-omega0)*self.dt)))
        den = np.where(omega == omega0, 1, (1-np.exp(1j*(omega-omega0)*self.dt)))
        return num/den
    def hann_dtft(self,f,f0):
        omega = 2*np.pi*f
        omega0 = 2*np.pi*f0
        domega = 2*np.pi/self.N
        return np.where(f == f0, self.N,np.exp(1j*(omega-omega0)*self.N/2)*(
            0.5*np.sin((omega-omega0)*self.N/2)/np.sin((omega-omega0)/2)
            +0.25*np.sin((omega-omega0-domega)*self.N/2)/np.sin((omega-omega0-domega)/2)
            +0.25*np.sin((omega-omega0+domega)*self.N/2)/np.sin((omega-omega0+domega)/2)))
    def hann(self,n,f0):
        omega0 = 2*np.pi*f0
        return np.where(n < 0 or n > self.N,0,
            np.exp(-1j*omega0*n*self.dt) * 0.5 * (1 - np.cos(2*np.pi*n/self.N))
            )
    def rect(self,n,f0):
        omega0 = 2*np.pi*f0
        #return np.exp(-1j*omega0*(n-self.N/2)*self.dt)
        return np.where(n < 0 or n > self.N,0,np.exp(-1j*omega0*(n)*self.dt))
    
    def __call__(self,t):
        n = int(np.round((t)/self.dt))
        vec = self.rect(n,self.center_frequencies)
        return np.dot(vec,self.nodes)
    
    def func(self):
        def _f(t): 
            return self(t)
        return _f
    
    def estimate_impulse_response(self,H):
        # Use vandermonde matrix to calculate weights of each gaussian.
        # Each sinc is centered at each frequency point
        vandermonde = self.sinc(self.frequencies[:,np.newaxis],self.center_frequencies[np.newaxis,:])
        nodes = np.matmul(np.linalg.pinv(vandermonde),H)
        H_hat = np.matmul(vandermonde,nodes)
        l2_err = np.sum(np.abs(H-H_hat)**2/np.abs(H)**2)
        print(l2_err)

        from matplotlib import pyplot as plt
        temp = self.sinc(self.frequencies[:,np.newaxis],self.center_frequencies)
        i_hat = np.inner(nodes,temp)
        
        plt.figure(figsize=(6,4))
        plt.subplot(3,2,1)
        plt.semilogy(self.frequencies,np.abs(H)**2)
        plt.semilogy(self.frequencies,np.abs(i_hat)**2,'--')
        plt.xlabel('Freq')
        
        plt.subplot(3,2,2)
        plt.plot(self.frequencies,np.abs(H)**2)
        plt.plot(self.frequencies,np.abs(i_hat)**2,'--')
        plt.xlabel('Freq')
        
        '''plt.subplot(2,1,2)
        plt.plot(self.frequencies,np.unwrap(np.angle(H)))
        plt.plot(self.frequencies,np.unwrap(np.angle(i_hat)),'--')'''
        #plt.show()
        #quit()
        
        return nodes, l2_err