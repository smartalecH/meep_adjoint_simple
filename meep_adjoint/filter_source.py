import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt


class FilteredCustomSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,num_taps,dt,time_src,**kwargs):
        self.center_frequency=center_frequency
        self.frequencies=frequencies
        self.frequency_response=frequency_response
        self.num_taps=500
        self.dt=dt
        self.time_src=time_src
        f = self.func()

        self.current_time = None
        #quit()

        # initialize super
        super(FilteredCustomSource, self).__init__(src_func=f,center_frequency=self.center_frequency)

        # Set up a gaussian if the user fails to supply a custom envelope
        '''if time_src_func is None:
            gaussian = mp.gaussian_src_time(self.center_frequency, width, self.start_time, self.start_time + 2 * width * self.cutoff)
            self.time_src_func = gaussian.dipole
        else:
            self.time_src_func=time_src_func'''

        # calculate equivalent sample rate
        self.fs = 1/self.dt

        # estimate impulse response from frequency response
        self.estimate_impulse_response()

    def filter(self,t):
        # shift feedforward memory
        np.roll(self.memory, -1)
        if self.current_time is None or self.current_time != t:
            self.current_time = t
            self.memory[0] = self.time_src.swigobj.dipole(t)
            # calculate filter response
            self.current_y = np.dot(self.memory,self.taps)
        return self.current_y
    
    def estimate_impulse_response(self):
        # calculate band edges from target frequencies
        w = self.frequencies/(self.fs/2) * np.pi
        D = self.frequency_response
        #self.taps = lstsqrs(self.num_taps,w,D)
        self.taps = cheb(self.num_taps,w,D)

        # allocate filter memory taps
        self.memory = np.zeros(self.taps.shape,dtype=np.complex128)
    
    def func(self):
        def _f(t): 
            return self.filter(t)
        return _f



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