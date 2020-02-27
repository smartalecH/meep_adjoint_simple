import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def gauss_t(t,f0,fwidth):
    s = 5
    w = 1.0 / fwidth
    t0 = w * s
    amp = 1.0 / -2 * 1j * np.pi * f0
    return amp * np.exp(-(t-t0)**2/(2*w*w))*np.exp(-1j*2*np.pi*f0*(t-t0))
    

def gauss_f(f,f0,fwidth):
    s = 5
    w = 1.0 / fwidth
    t0 = w * s
    omega = 2.0 * np.pi * f
    omega0 = 2.0 * np.pi * f0
    delta = (omega - omega0) * w
    return w * np.exp(1j*omega*t0) * np.exp(-0.5 * delta * delta)

def rvf(freqs,h_desired):
    vandermonde = np.zeros((freqs.size,freqs.size),dtype=np.complex128)
    for ri, rf in enumerate(freqs):
        for ci, cf in enumerate(freqs):
            vandermonde[ri,ci] = gauss_f(rf,cf,fwidth_example)
    
    nodes = np.matmul(np.linalg.pinv(vandermonde),h_desired)
    h_hat = np.matmul(vandermonde,nodes)
    l2_err = np.sum(np.abs(h_desired-h_hat))

    print("l2 error: ",l2_err)
    return nodes

fwidth_example =  1/1.55*0.05


dec = 50
f = np.linspace(1/1.6, 1/1.5, 1000)
t0 = 200
fc = 1/1.55
i = np.exp(-1e4*(f-fc)**2) * np.exp(1j*2*np.pi*t0*(f-fc))

nodes = rvf(f[0::dec],i[0::dec])
temp = gauss_f(f[:,np.newaxis],f[0::dec].T,fwidth_example)
i_hat = np.inner(nodes,temp)

plt.figure()
plt.plot(f,np.real(i_hat))
plt.plot(f[0::dec],np.real(i[0::dec]),'o')


N = 6e4
t = np.linspace(0,4000,N)
dt = t[2] - t[1]
t_resp = np.dot(gauss_t(t[:,np.newaxis],f[0::dec],fwidth_example),nodes)

plt.figure()
plt.plot(t,np.imag(t_resp))

dft_h = np.fft.fftshift(np.fft.fft(t_resp))
dft_freq = np.fft.fftshift(np.fft.fftfreq(dft_h.size, d=(t[1]-t[0])))
#dft_freq = np.linspace(0,1,N)
print(dft_freq)

plt.figure()
plt.subplot(2,1,1)
plt.plot(dft_freq,np.real(dft_h))
plt.plot(f,np.real(i_hat))
#plt.xlim(-1/1.5,-1/1.6)

plt.subplot(2,1,2)
plt.plot()
plt.show()
