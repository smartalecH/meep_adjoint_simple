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
        self.forward = 1 if forward else 0
        self.direction = None
        self.k0 = k0
        self.eval = None
        return
    
    def register_monitors(self):
        self.monitor = self.sim.add_flux(self.fcen,self.df,self.nf,mp.FluxRegion(center=self.volume.center,size=self.volume.size))
        self.direction = self.monitor.normal_direction
        return self.monitor
    
    def _calc_vgrp(self):
        EigenmodeData = self.sim.get_eigenmode(self.fcen, self.direction, self.volume, self.mode, mp.Vector3())
        EH_TRANSVERSE    = [[mp.Ey, mp.Ez, mp.Hy, mp.Hz],
                            [mp.Ez, mp.Ex, mp.Hz, mp.Hx],
                            [mp.Ex, mp.Ey, mp.Hx, mp.Hy] ]
        
        x,y,z,w = self.sim.get_array_metadata(dft_cell=self.monitor)

        Emy = np.zeros((len(x),len(y),len(z)),dtype=np.complex128)
        Emz = np.zeros((len(x),len(y),len(z)),dtype=np.complex128)
        Hmy = np.zeros((len(x),len(y),len(z)),dtype=np.complex128)
        Hmz = np.zeros((len(x),len(y),len(z)),dtype=np.complex128)

        mcomps = [Emy,Emz,Hmy,Hmz]
        for ic,c in enumerate(EH_TRANSVERSE[self.direction]):
            for ix,px in enumerate(x):
                for iy,py in enumerate(y):
                    for iz,pz in enumerate(z):
                        mcomps[ic][ix,iy,iz] = EigenmodeData.amplitude(mp.Vector3(px,py,pz),c)
        
        # FIXME change for 3d resolution
        MO1 = -np.sum(np.conj(Hmy) * Emz - np.conj(Hmz) * Emy,axis=None) * (1/self.sim.resolution)
        MO2 = np.sum(np.conj(Emy) * Hmz - np.conj(Emz) * Hmy,axis=None) * (1/self.sim.resolution)
        vgrp = (MO1 + MO2)/2
        return vgrp
    
    def _calculate_scalars(self):
        if self.eval is None:
            raise RuntimeError("The eigenmode coefficients must be calculated before trying to scale the adjoint source.")
        # FIXME implement cscale using actual vgrp
        cscale = np.abs(np.sqrt(1/np.abs(self.vgrp)))
        # FIXME resolution for 3d sims and planar monitors
        # FIXME grab fcen from self.eval
        # TODO single valued scaling -- it would be better to fit to a pade approximant and apodize in time domain
        scale = 2 / self.sim.resolution * cscale * 1/np.sqrt(self.forward_power)* 2 / self.sim.resolution
        
        return scale
    
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
        scale = self._calculate_scalars() * dJ * 2 * np.pi * 1j* self.time_src.frequency
        print(scale)
        print(dJ)
        quit()
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
        # record simulation's forward power for later scaling FIXME find better way to do this in case sources are complicated...
        self.forward_power = self.sim.sources[0].eig_power(self.time_src.frequency)

        # record eigenmode coefficients for scaling
        ob = self.sim.get_eigenmode_coefficients(self.monitor,[self.mode])
        self.eval = np.squeeze(ob.alpha[:,:,self.direction])

        # record all freqs of interest # FIXME allow for broadband objectives
        self.freqs = mp.get_eigenmode_freqs(self.monitor)

        # pull group velocity # FIXME use native get_eigenmode_coefficients once bug is fixed
        self.vgrp = self._calc_vgrp()#np.squeeze(ob.vgrp)
        return self.eval