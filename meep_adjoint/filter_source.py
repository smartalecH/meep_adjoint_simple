import meep as mp
import numpy as np
from scipy import signal
from meep import CustomSource


class FilteredCustomSource(CustomSource):
    def __init__(self,center_frequency,frequencies,frequency_response,num_taps,dt,time_src_func=None,cutoff=5.0,**kwargs):
        self.center_frequency=center_frequency
        self.frequencies=frequencies
        self.frequency_response=frequency_response
        self.num_taps=num_taps
        self.dt=dt
        self.cutoff = cutoff

        # initialize super
        super(FilteredCustomSource, self).__init__(src_func=self.filter,center_frequency=self.center_frequency,**kwargs)

        # Set up a gaussian if the user fails to supply a custom envelope
        if time_src_func is None:
            width = np.max(self.frequencies) - np.min(self.frequencies)
            gaussian = mp.gaussian_src_time(self.center_frequency), width, self.start_time, self.start_time + 2 * self.width * self.cutoff)
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
        df = self.fwidth/self.nf
        edge0 = self.frequencies[0] - df
        edges = self.frequencies + df
        edges = np.insert(edges,0,edge0) / self.fs/2

        # estimate real part using PM
        real_taps = signal.remez(self.num_taps, edges, np.real(self.frequency_response), fs=self.fs)

        # estimate imag part using PM
        imag_taps = signal.remez(self.num_taps, edges, np.imag(self.frequency_response), type='hilbert', fs=self.fs)

        # sum FIR together to get final response
        self.taps = real_taps + imag_taps

        # allocate filter memory taps
        self.memory = np.zeros(self.taps.shape)