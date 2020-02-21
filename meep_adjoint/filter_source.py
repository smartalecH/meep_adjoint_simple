import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource
import matplotlib.pyplot as plt


class FilteredCustomSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,num_taps,dt,width,time_src_func=None,cutoff=5.0,**kwargs):
        self.center_frequency=center_frequency
        self.frequencies=frequencies
        self.frequency_response=frequency_response
        self.num_taps=10
        self.dt=dt
        self.cutoff = cutoff
        f = self.func()

        # initialize super
        super(FilteredCustomSource, self).__init__(src_func=f,center_frequency=self.center_frequency)

        # Set up a gaussian if the user fails to supply a custom envelope
        if time_src_func is None:
            gaussian = mp.gaussian_src_time(self.center_frequency, width, self.start_time, self.start_time + 2 * width * self.cutoff)
            self.time_src_func = gaussian.dipole
        else:
            self.time_src_func=time_src_func

        # calculate equivalent sample rate
        self.fs = 1/self.dt

        # estimate impulse response from frequency response
        self.estimate_impulse_response()
    
    def filter(self,t):
        # shift feedforward memory
        np.roll(self.memory, 1)
        self.memory[0] = t

        # calculate filter response
        return np.dot(self.memory,self.taps)
    
    def estimate_impulse_response(self):
        '''
        Reference: https://dspguru.com/dsp/tricks/using-parks-mcclellan-to-design-non-linear-phase-fir-filter/
        '''

        # calculate band edges from target frequencies
        df = self.frequencies[1] - self.frequencies[0]
        edges = np.array([[a-df/3,a+df/3] for a in self.frequencies]).flatten()
        edges = np.concatenate(([0, edges[0]-df/3],edges,[edges[-1]+df/3,self.fs/2]))
        gain = np.concatenate(([0],self.frequency_response,[0]))

        '''# estimate real part using PM
        real_taps = signal.remez(self.num_taps, edges, np.real(gain), fs=self.fs,grid_density=32)
        # estimate imag part using PM
        imag_taps = signal.remez(self.num_taps, edges, np.imag(gain), type='hilbert', fs=self.fs)

        # sum FIR together to get final response
        self.taps = real_taps + imag_taps'''
        w = self.frequencies/(self.fs/2) * np.pi
        D = self.frequency_response
        W = np.ones(D.shape)
        #(a,b) = eqnerror(self.num_taps,self.num_taps,w,D,W,iter=1)
        lstsqrs(self.num_taps,w,D)
        quit()

        w,h = signal.freqz(b,a)
        print(h)
        #print(np.imag(self.frequency_response))

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(0.5*self.fs*w/np.pi,np.abs(h)**2)
        plt.plot(self.frequencies,np.abs(self.frequency_response)**2,'o')
        #plt.xlim(self.frequencies[0],self.frequencies[-1])

        plt.subplot(2,1,2)
        plt.plot(0.5*self.fs*w/np.pi,np.unwrap(np.angle(h)))
        plt.plot(self.frequencies,np.unwrap(np.angle(self.frequency_response)),'o')
        #plt.xlim(self.frequencies[0],self.frequencies[-1])

        plt.show()
        quit()
        # allocate filter memory taps
        self.memory = np.zeros(self.taps.shape)
    def func(self):
        def _f(t): 
            return self(t)
        return _f



#function [a,b] = eqnerror(M,N,w,D,W,iter);

def lstsqrs(num_taps,freqs,h_desired):
    ''' 
    freqs = [0,pi]
    '''
    n_freqs = freqs.size
    vandermonde = np.zeros((n_freqs,num_taps),dtype=np.complex128)
    for fi, f in enumerate(freqs):
        for ci in range(num_taps):
            vandermonde[fi,ci] = np.exp(-1j*ci*f)
    
    a = np.matmul(np.linalg.pinv(vandermonde), h_desired)
    print(np.rad2deg(np.angle(vandermonde[:,3]/vandermonde[:,2])))
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