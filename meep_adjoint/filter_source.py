import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt


class FilteredSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,time_src,min_err=1e-6):
        self.center_frequency=center_frequency
        self.frequencies=frequencies

        signal_fourier_transform = np.array([time_src.fourier_transform(f) for f in self.frequencies])
        self.frequency_response= np.conjugate(signal_fourier_transform)
        self.time_src=time_src
        self.current_time = None
        self.min_err = min_err

        f = self.func()

        # initialize super
        super(FilteredSource, self).__init__(src_func=f,center_frequency=self.center_frequency)

        # estimate impulse response from frequency response
        self.estimate_impulse_response()
        

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
        def rbf_l2(fwidth):
            vandermonde = np.zeros((self.frequencies.size,self.frequencies.size),dtype=np.complex128)
            for ri, rf in enumerate(self.frequencies):
                for ci, cf in enumerate(self.frequencies):
                    vandermonde[ri,ci] = self.gauss_f(rf,cf,fwidth)
            
            nodes = np.matmul(np.linalg.pinv(vandermonde),self.frequency_response)
            h_hat = np.matmul(vandermonde,nodes)
            l2_err = np.sum(np.abs(self.frequency_response-h_hat)**2)
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
        plt.semilogy(self.frequencies,np.abs(self.frequency_response)**2)
        plt.semilogy(self.frequencies,np.abs(i_hat)**2)

        plt.subplot(2,1,2)
        plt.plot(self.frequencies,np.unwrap(np.angle(self.frequency_response)))
        plt.plot(self.frequencies,np.unwrap(np.angle(i_hat)))
    
    def gauss_t(self,t,f0,fwidth):
        s = 10
        w = 1.0 / fwidth
        t0 = w * s
        tt = (t - t0)
        amp = 1/np.pi  # compensate for meep's dtft scaling
        return amp * np.exp(-tt * tt / (2 * w * w))*np.exp(1j*2 * np.pi * f0 * tt)
    
    def gauss_f(self,f,f0,fwidth):
        s = 10
        w = 1.0 / fwidth
        t0 = w * s
        omega = 2.0 * np.pi * f
        omega0 = 2.0 * np.pi * f0
        delta = (omega - omega0) * w
        return w * np.exp(-1j*omega*t0) * np.exp(-0.5 * delta * delta) / (2*np.pi)



#function [a,b] = eqnerror(M,N,w,D,W,iter);

def cheb(num_taps,freqs,h_desired):
    deg = 1000
    pm = np.polynomial.chebyshev.Chebyshev.fit(freqs,np.abs(h_desired),deg,[0,np.pi])
    pp = np.polynomial.chebyshev.Chebyshev.fit(freqs,np.unwrap(np.angle(h_desired)),deg,[0,np.pi])
    #om = np.linspace(freqs[0],freqs[-1],1e3)
    om = np.linspace(0,np.pi,1e5)
    fit = pm(om)*np.exp(1j*pp(om))
    print(np.pi/(freqs[2]-freqs[1]))


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(om,np.real(fit))
    plt.plot(freqs,np.real(h_desired),'o')

    plt.subplot(2,1,2)
    plt.plot(om,np.imag(fit))
    plt.plot(freqs,np.imag(h_desired),'o')

    plt.show()
    quit()

def lstsqrs(num_taps,freqs,h_desired):
    ''' 
    Current solution:
    1. Build vandermonde/gram matrix of the dft basis functions
    2. Perform least squares fit using psuedoinverse (optimal in l2 sense)
    3. Coefficients are filter taps in time domain (since the DFT basis functions are just filter tap time delays)

    We could gradually increase taps until an error criteria is met
    freqs = [0,pi]

    Another possible solution:
    1. Try fitting to a chebyshev polynomial of arbitrary degree
    2. Sample in DFT domain according to number desired taps
    3. IFFT to get time domain filter taps
    '''
    n_freqs = freqs.size
    vandermonde = np.zeros((n_freqs,num_taps),dtype=np.complex128)
    for iom, om in enumerate(freqs):
        for it in range(num_taps):
            vandermonde[iom,it] = np.exp(-1j*it*om)
    
    
    a = np.matmul(np.linalg.pinv(vandermonde), h_desired)
    _, h_hat = signal.freqz(a,worN=freqs)
    l2_error = np.sqrt(np.sum(np.abs(h_hat - h_desired)**2))

    '''print(l2_error)
    
    worN = np.linspace(freqs[0],freqs[-1],500)
    w, h = signal.freqz(a,worN=worN)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,np.real(h))
    plt.plot(freqs,np.real(h_desired),'o')
    #plt.xlim(freqs[0],freqs[-1])
    plt.ylim(-.01,.01)

    plt.subplot(2,1,2)
    plt.plot(w,np.imag(h))
    plt.plot(freqs,np.imag(h_desired),'o')
    plt.ylim(-.01,.01)

    plt.show()

    quit()'''
    
    return a
def eqnerror(M,N,w,D,W,iter=1):
    '''
    % [a,b] = eqnerror(M,N,w,D,W,iter);
    %
    % IIR filter design using equation error method
    %
    % if the input argument 'iter' is specified and if iter > 1, then
    % the equation error method is applied iteratively trying to
    % determine the true L2 solution of the nonlinear IIR design problem
    %
    % M     numerator order
    % N     denominator order
    % a     denominator coefficients (length N+1), a(1) = 1
    % b     numerator coefficients (length M+1)
    % w     frequency vector in [0,pi], where pi is Nyquist
    % D     desired complex frequency response at frequencies w
    % W     weight vector defined at frequencies w
    % iter  optional; number of iterations for non-linear solver
    %
    % author: Mathias C. Lang, 2016-02-28
    % mattsdspblog@gmail.com
    '''
    w = np.squeeze(np.asarray(w))
    D = np.squeeze(np.asarray(D))
    W = np.squeeze(np.asarray(W))
    L = w.size

    if (max(w) > np.pi or min(w) < 0):  raise ValueError('w must be in [0,pi]') 
    if D.size != L: raise ValueError('D and w must have the same lengths.') 
    if W.size != L: raise ValueError('W and w must have the same lengths.') 

    W0=W
    D0=D

    left = -np.tile(D0[:, np.newaxis],(1,N)) * np.exp(-1j*np.outer(w,np.arange(1,N+1)))
    right = np.exp(-1j*np.outer(w,np.arange(0,M+1)))
    A0 = np.concatenate((left,right),axis=1)
    #A0 = [-D0(:,ones(N,1)).*exp(-1i*w*(1:N)), exp(-1i*w*(0:M))];
    den = np.ones((L,))

    for k in range(iter):
        W = W0/np.abs(den)
        A = A0 * np.tile(W[:, np.newaxis],(1,M+N+1))
        D = D0*W
        x = np.matmul(np.linalg.pinv(np.concatenate((np.real(A),np.imag(A)))), np.concatenate((np.real(D),np.imag(D))))
        a = np.concatenate(([1],x[0:N]))
        b = x[N:]
        _, den = signal.freqz(a,1,w)
    return (a,b)