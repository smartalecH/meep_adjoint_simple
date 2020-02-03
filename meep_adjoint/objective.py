"""Handling of objective functions and objective quantities."""

from abc import ABC, abstractmethod
import numpy as np
import meep as mp

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
    def __init__(self,sim,volume,fcen,df,nf,mode,time_src,forward=True,k0=None):
        '''
        time_src ............... time dependence of source
        '''
        self.sim = sim
        self.volume=volume
        self.fcen=fcen
        self.df=df
        self.nf=nf
        self.mode=mode
        self.time_src = time_src
        self.forward = 0 if forward else 1
        self.direction = None
        self.k0 = k0
        self.eval = None
        return
    
    def register_monitors(self):
        self.monitor = self.sim.add_flux(self.fcen,self.df,self.nf,mp.FluxRegion(center=self.volume.center,size=self.volume.size))
        self.direction = self.monitor.normal_direction
        return self.monitor
    
    def place_adjoint_source(self,dJ):
        '''
        dJ ........ the user needs to pass the dJ/dMonitor evaluation
        '''
        
        # determine starting point for reverse mode eigenmode source
        direction_scalar = 1 if self.direction else -1
        if self.k0 is None:
            if self.direction == 0:
                k0 = direction_scalar * mp.Vector3(x=1)
            elif self.direction == 1:
                k0 = direction_scalar * mp.Vector3(y=1)
            elif self.direction == 2:
                k0 == direction_scalar * mp.Vector3(z=1)
        else:
            k0 = direction_scalar * self.k0
        
        # get scaling factor 
        # FIXME currently assumes evaluating frequency factor at center is good enough
        # NOTE multiply j*2*pi*f after adjoint simulation since it's a simple scalar that is inherently freq dependent
        da_dE = 0.5*(1 / self.sim.resolution * 1 / self.sim.resolution * self.cscale * 1/np.sqrt(self.forward_power))
        scale = da_dE * dJ      

        # generate source
        self.source = mp.EigenModeSource(self.time_src,
                    eig_band = self.mode,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=k0,
                    size = self.volume.size,
                    amplitude = scale,
                    center=self.volume.center)
        
        # record adjoint power at each freq
        self.adjoint_power = self.source.eig_power(self.time_src.frequency)
        
        return self.source

    def __call__(self):
        # record simulation's forward power for later scaling 
        # FIXME find better way to do this in case sources are complicated...
        self.forward_power = self.sim.sources[0].eig_power(self.time_src.frequency)

        # record eigenmode coefficients for scaling
        ob = self.sim.get_eigenmode_coefficients(self.monitor,[self.mode])
        self.eval = np.squeeze(ob.alpha[:,:,self.forward])

        # record all freqs of interest
        # FIXME allow for broadband objectives
        self.freqs = mp.get_eigenmode_freqs(self.monitor)

        # pull scaling factor 
        # FIXME only record center frequency
        self.cscale = np.squeeze(ob.cscale)

        return self.eval