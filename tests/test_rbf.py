import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

def gauss_t(t,f0,fwidth):
    s = 10
    w = 1.0 / fwidth
    t0 = w * s
    tt = (t - t0)
    amp = 1.0 #/ (-2 * 1j * np.pi * f0)
    return amp * np.exp(-tt * tt / (2 * w * w))*np.exp(1j*2 * np.pi * f0 * tt)

def gauss_f(f,f0,fwidth):
    s = 10
    w = 1.0 / fwidth
    t0 = w * s
    omega = 2.0 * np.pi * f
    omega0 = 2.0 * np.pi * f0
    delta = (omega - omega0) * w
    return w * np.exp(-1j*omega*t0) * np.exp(-0.5 * delta * delta) / (2*np.pi)

def rvf(freqs,h_desired,num_taps):
    #tap_freqs = np.linspace(freqs[0],freqs[-1],num_taps)
    tap_freqs = freqs
    vandermonde = np.zeros((freqs.size,freqs.size),dtype=np.complex128)
    for ri, rf in enumerate(freqs):
        for ci, cf in enumerate(freqs):
            vandermonde[ri,ci] = gauss_f(rf,cf,fwidth_example)
    
    nodes = np.matmul(np.linalg.pinv(vandermonde),h_desired)
    h_hat = np.matmul(vandermonde,nodes)
    l2_err = np.sum(np.abs(h_desired-h_hat))

    print("l2 error: ",l2_err)
    return nodes, tap_freqs

fwidth_example =  1/1.55*0.05 * 0.2


dec = 10
f = np.linspace(1/1.6, 1/1.5, 1000)
t0 = 200
fc = 1/1.55
i = np.exp(-1e4*(f-fc)**2) * np.exp(1j*2*np.pi*t0*(f-fc))

nodes, tap_freqs = rvf(f[0::dec],i[0::dec],num_taps=20)
temp = gauss_f(f[:,np.newaxis],tap_freqs,fwidth_example)
i_hat = np.inner(nodes,temp)

plt.figure()
plt.plot(f,np.real(i_hat))
plt.plot(f[0::dec],np.real(i[0::dec]),'o')


N = 10e4
t = np.linspace(0,4000,N)
dt = t[2] - t[1]
t_resp = np.dot(gauss_t(t[:,np.newaxis],tap_freqs,fwidth_example),nodes)
#t_resp = (gauss_t(t,1/1.55,fwidth_example))


plt.figure()
plt.plot(t,np.imag(t_resp))

dft_h = np.fft.fftshift(np.fft.fft(t_resp,norm='ortho'))
dft_freq = np.fft.fftshift(np.fft.fftfreq(dft_h.size, d=dt))


plt.figure()
plt.subplot(2,1,1)
plt.plot(dft_freq,np.real(dft_h))
plt.plot(f,np.real(i),'--')
plt.plot(f[0::dec],np.real(i[0::dec]),'o')
plt.xlim(1/1.6,1/1.5)
#plt.ylim(-1,1)

plt.subplot(2,1,2)
plt.plot(dft_freq,np.imag(dft_h))
plt.plot(f,np.imag(i),'--')
plt.plot(f[0::dec],np.imag(i[0::dec]),'o')
plt.xlim(1/1.6,1/1.5)
#plt.ylim(-1,1)
plt.plot()
plt.show()
