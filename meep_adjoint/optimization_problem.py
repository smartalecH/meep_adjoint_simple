import meep as mp
from autograd import grad
import numpy as np
from collections import namedtuple

Grid = namedtuple('Grid', ['x', 'y', 'z', 'w'])

class OptimizationProblem(object):
    """Top-level class in the MEEP adjoint module.

    Intended to be instantiated from user scripts with mandatory constructor
    input arguments specifying the data required to define an adjoint-based
    optimization.

    The class knows how to do one basic thing: Given an input vector
    of design variables, compute the objective function value (forward
    calculation) and optionally its gradient (adjoint calculation).
    This is done by the __call__ method. The actual computations
    are delegated to a hierarchy of lower-level classes, of which
    the uppermost is TimeStepper.

    """

    def __init__(self, 
                simulation,
                objective_function,
                objective_arguments,
                basis,
                fcen,
                df=0,
                nf=1,
                time=1200
                 ):

        self.sim = simulation
        self.objective_function = objective_function
        self.objective_arguments = objective_arguments
        self.basis = basis
        self.design_region = self.basis.volume
        
        self.fcen = fcen
        self.df = df
        self.nf = nf
        self.freq_min = self.fcen - self.df/2
        self.freq_max = self.fcen + self.df/2

        # TODO add dynamic method that checks for convergence
        self.time=time
        self.num_design_params = self.basis.num_design_params
        
        # store sources for finite difference estimations
        self.forward_sources = self.sim.sources     

        # --------------------------------------------------------- #
        # Prepare forward run
        # --------------------------------------------------------- #

        # register user specified monitors
        for m in self.objective_arguments:
            m.register_monitors(self.fcen,self.df,self.nf)

        # register design region
        self.design_region_monitor = self.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],self.freq_min,self.freq_max,self.nf,where=self.design_region,yee_grid=False)

        # store design region voxel parameters
        self.design_grid = Grid(*self.sim.get_array_metadata(dft_cell=self.design_region_monitor))

    def __call__(self, rho_vector=None, need_value=True, need_gradient=True):
        """Evaluate value and/or gradient of objective function.
        """
        if rho_vector is not None:
            self.update_design(rho_vector=rho_vector)

        # Run forward run
        # FIXME check if we actually need a forward run
        self.forward_run()

        # Run adjoint simulation
        # FIXME check if we actually need an adjoint run
        self.adjoint_run()

        # calculate gradient
        self.calculate_gradient()

        return self.f0, self.gradient

    def get_fdf_funcs(self):
        """construct callable functions for objective function value and gradient

        Returns
        -------
        2-tuple (f_func, df_func) of standalone (non-class-method) callables, where
            f_func(beta) = objective function value for design variables beta
           df_func(beta) = objective function gradient for design variables beta
        """

        def _f(x=None):
            (fq, _) = self.__call__(beta_vector = x, need_gradient = False)
            return fq[0]

        def _df(x=None):
            (_, df) = self.__call__(need_value = False)
            return df

        return _f, _df

    def forward_run(self):

        # Forward run
        self.sim.run(until=self.time)

        # record objective quantities from user specified monitors
        self.results_list = []
        for m in self.objective_arguments:
            self.results_list.append(m())

        # evaluate objective
        self.f0 = self.objective_function(*self.results_list)

        # Store forward fields in array (x,y,z,field_components,frequencies)
        # FIXME allow for multiple design regions
        self.d_E = np.zeros((len(self.design_grid.x),len(self.design_grid.y),len(self.design_grid.z),3,self.nf),dtype=np.complex128)
        for f in range(self.nf):
            for ic, c in enumerate([mp.Ex,mp.Ey,mp.Ez]):
                self.d_E[:,:,:,ic,f] = np.atleast_3d(self.sim.get_dft_array(self.design_region_monitor,c,f))

    def adjoint_run(self):
        # Prepare adjoint run
        self.sim.reset_meep()

        self.adjoint_sources = []
        for mi, m in enumerate(self.objective_arguments):
            dJ = grad(self.objective_function,mi)(*self.results_list) # get gradient of objective w.r.t. monitor
            self.adjoint_sources.append(m.place_adjoint_source(dJ)) # place the appropriate adjoint sources

        self.sim.change_sources(self.adjoint_sources)

        # reregsiter design flux
        # FIXME cleanup design region input
        # TODO use yee grid directly 
        self.design_region_monitor = self.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],self.freq_min,self.freq_max,self.nf,where=self.design_region,yee_grid=False)

        # Adjoint run
        # TODO make more dynamic
        self.sim.run(until=self.time)

        # Store adjoint fields in array (x,y,z,field_components,frequencies)
        # FIXME allow for multiple design regions
        self.a_E = np.zeros((len(self.design_grid.x),len(self.design_grid.y),len(self.design_grid.z),3,self.nf),dtype=np.complex128)
        for f in range(self.nf):
            for ic, c in enumerate([mp.Ex,mp.Ey,mp.Ez]):
                self.a_E[:,:,:,ic,f] = np.atleast_3d(self.sim.get_dft_array(self.design_region_monitor,c,f))
        
        # store frequencies
        self.frequencies = np.array(mp.get_flux_freqs(self.design_region_monitor))

    def calculate_gradient(self):

        # calculate frequency scalar
        adjoint_powers = 0
        for oa in self.objective_arguments:
            adjoint_powers += oa.adjoint_power
        frequency_scalar = 1j*2*np.pi*self.frequencies * 1 / np.sqrt(adjoint_powers)
        print(adjoint_powers)

        #quit()

        
        self.gradient = self.basis.gradient(self.d_E, self.a_E, frequency_scalar, self.design_grid)
        # FIXME record run stats for checking later
    
    def calculate_fd_gradient(self,num_gradients=1,db=1e-4):
        '''
        Estimate central difference gradients.
        '''

        if num_gradients > self.num_design_params:
            raise ValueError("The requested number of gradients must be less than or equal to the total number of design parameters.")

        # cleanup simulation object
        self.sim.reset_meep()
        self.sim.change_sources(self.forward_sources)

        # preallocate result vector
        fd_gradient = 0*np.ones((self.num_design_params,))

        # randomly choose indices to loop estimate
        fd_gradient_idx = np.random.choice(self.num_design_params,num_gradients,replace=False)

        for k in fd_gradient_idx:
            
            b0 = np.ones((self.num_design_params,))
            b0[:] = self.basis.rho_vector
            # -------------------------------------------- #
            # left function evaluation
            # -------------------------------------------- #
            self.sim.reset_meep()
            
            # assign new design vector
            b0[k] -= db
            self.basis.set_rho_vector(b0)
            
            # initialize design monitors
            for m in self.objective_arguments:
                m.register_monitors(self.fcen,self.df,self.nf)
            
            # run simulation FIXME make dyanmic time
            self.sim.run(until=self.time)
            
            # record final objective function value
            results_list = []
            for m in self.objective_arguments:
                results_list.append(m())
            fm = self.objective_function(*results_list)

            # -------------------------------------------- #
            # right function evaluation
            # -------------------------------------------- #
            self.sim.reset_meep()

            # assign new design vector
            b0[k] += 2*db # central difference rule...
            self.basis.set_rho_vector(b0)

            # initialize design monitors
            for m in self.objective_arguments:
                m.register_monitors(self.fcen,self.df,self.nf)
            
            # run simulation FIXME make dyanmic time
            self.sim.run(until=self.time)
            
            # record final objective function value
            results_list = []
            for m in self.objective_arguments:
                results_list.append(m())
            fp = self.objective_function(*results_list)

            # -------------------------------------------- #
            # estimate derivative
            # -------------------------------------------- #
            fd_gradient[k] = (fp - fm) / (2*db)
        
        return fd_gradient, fd_gradient_idx
    
    def update_design(self, rho_vector):
        """Update the design permittivity function.
        """
        self.basis.set_rho_vector(rho_vector)

    def visualize(self, id=None, pmesh=False):
        """Produce a graphical visualization of the geometry and/or fields,
           as appropriately autodetermined based on the current state of
           progress.
        """
        
        if self.stepper.state=='reset':
            self.stepper.prepare('forward')

        bs = self.basis
        mesh = bs.fs.mesh() if (hasattr(bs,'fs') and hasattr(bs.fs,'mesh')) else None

        fig = plt.figure(num=id) if id else None

        self.sim.plot2D()
        if mesh is not None and pmesh:
            plot_mesh(mesh)

def plot_mesh(mesh,lc='g',lw=1):
    """Helper function. Invoke FENICS/dolfin plotting routine to plot FEM mesh"""
    try:
        import dolfin as df
        df.plot(mesh, color=lc, linewidth=lw)
    except ImportError:
        warnings.warn('failed to import dolfin module; omitting FEM mesh plot')