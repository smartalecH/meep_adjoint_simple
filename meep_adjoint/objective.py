"""Handling of objective functions and objective quantities."""

from abc import ABC, abstractmethod
import numpy as np
import meep as mp
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs

class ObjectiveQuantitiy(ABC):
    @abstractmethod
    def __init__(self):
        return
    @abstractmethod
    def register_monitors(self):
        return
    @abstractmethod
    def place_adjoint_source(self):
        return

class EigenmodeCoefficient(ObjectiveQuantitiy):
    def __init__(self,sim,volume,mode,forward=True,k0=None):
        '''
        time_src ............... time dependence of source
        '''
        self.sim = sim
        self.volume=volume
        self.mode=mode
        self.forward = 0 if forward else 1
        self.normal_direction = None
        self.k0 = k0
        self.eval = None
        return
    
    def register_monitors(self,fcen,df,nf):
        self.fcen=fcen
        self.df=df
        self.nf=nf

        self.monitor = self.sim.add_flux(self.fcen,self.df,self.nf,mp.FluxRegion(center=self.volume.center,size=self.volume.size))
        self.normal_direction = self.monitor.normal_direction
        return self.monitor
    
    def place_adjoint_source(self,dJ):
        '''
        dJ ........ the user needs to pass the dJ/dMonitor evaluation
        '''
        dJ = np.atleast_1d(dJ)
        # determine starting kpoint for reverse mode eigenmode source
        direction_scalar = 1 if self.forward else -1
        if self.k0 is None:
            if self.normal_direction == 0:
                k0 = direction_scalar * mp.Vector3(x=1)
            elif self.normal_direction == 1:
                k0 = direction_scalar * mp.Vector3(y=1)
            elif self.normal_direction == 2:
                k0 == direction_scalar * mp.Vector3(z=1)
        else:
            k0 = direction_scalar * self.k0
        
        # generate source
        self.source = mp.EigenModeSource(self.time_src,
                    eig_band = self.mode,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=k0,
                    size = self.volume.size,
                    center=self.volume.center)
        
        # -------------------------------------- #
        # Get scaling factor 
        # -------------------------------------- #

        self.adjoint_power = np.array([self.source.eig_power(f) for f in self.freqs])

        # FIXME currently assumes evaluating frequency factor at center is good enough. Future
        # implementations should convolve the source with the time dependent response (or something similar)
        # NOTE multiply j*2*pi*f after adjoint simulation since it's a simple scalar that is inherently freq dependent
        
        da_dE = 0.5*(1/self.sim.resolution * 1/self.sim.resolution * self.cscale)
        scale = da_dE * dJ[self.fcen_idx]# * 1/np.sqrt(self.adjoint_power)
        
        # scale the adjoint source appropriately
        #self.source.amplitude=scale
        self.scale_experiment = da_dE * dJ * 1/np.sqrt(self.adjoint_power)
        return self.source

    def __call__(self):
        # record simulation's forward power for later scaling 
        # TODO find better way to do this in case sources are complicated...
        # TODO allow for multiple input sources
        self.forward_power = self.sim.sources[0].eig_power(self.fcen)
        self.time_src = self.sim.sources[0].src

        # record eigenmode coefficients for scaling
        ob = self.sim.get_eigenmode_coefficients(self.monitor,[self.mode])
        self.eval = np.squeeze(ob.alpha[:,:,self.forward])

        # record all freqs of interest
        self.freqs = np.atleast_1d(mp.get_eigenmode_freqs(self.monitor))

        # get f0 frequency index
        self.fcen_idx = np.argmin(np.abs(self.freqs-self.fcen))

        # pull scaling factor 
        self.cscale = ob.cscale

        return self.eval