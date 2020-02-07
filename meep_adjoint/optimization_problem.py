"""OptimizationProblem is the top-level class exported by the meep.adjoint module.
"""
import meep as mp
from autograd import grad
import numpy as np


# OptimizationRegion?????


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
                design_function,
                basis,
                fcen,
                df=0,
                nf=1,
                time=1200
                 ):

        self.sim = simulation
        self.objective_function = objective_function
        self.objective_arguments = objective_arguments
        # FIXME better way to add design functions and basis functions... maybe with an object?
        self.design_function = design_function
        self.basis = basis
        self.design_region = self.basis.domain
        # FIXME proper way to add freqs
        self.fcen = fcen
        self.df = df
        self.nf = nf
        # record convergence time
        # FIXME add dynamic method that checks for convergence
        self.time=time
        self.num_design_params = self.basis.num_design_params
        
        # store sources for finite difference estimations
        self.forward_sources = self.sim.sources     

        # --------------------------------------------------------- #
        # Prepare forward run
        # --------------------------------------------------------- #
        # register user specified monitors
        # FIXME do we actually need to store the monitors?
        self.mon_list = []
        for m in self.objective_arguments:
            self.mon_list.append(m.register_monitors())

        # register design region
        self.mon_list.append(self.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],self.fcen,self.fcen,1,where=self.design_region,yee_grid=False))

    def __call__(self, beta_vector=None, need_value=True, need_gradient=True):
        """Evaluate value and/or gradient of objective function.
        """
        if beta_vector is not None:
            self.update_design(beta_vector=beta_vector)

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

        # --------------------------------------------------------- #
        # Forward run
        # --------------------------------------------------------- #
        self.sim.run(until=self.time)

        # --------------------------------------------------------- #
        # Process and store results
        # --------------------------------------------------------- #
        # record objective quantities from user specified monitors
        self.results_list = []
        for m in self.objective_arguments:
            self.results_list.append(m())

        # evaluate objective
        self.f0 = self.objective_function(*self.results_list)

        # record fields in design region (last entry of monitor list)
        # FIXME allow for multiple design regions
        self.d_Ex = self.sim.get_dft_array(self.mon_list[-1],mp.Ex,0)
        self.d_Ey = self.sim.get_dft_array(self.mon_list[-1],mp.Ey,0)
        self.d_Ez = self.sim.get_dft_array(self.mon_list[-1],mp.Ez,0)

    def adjoint_run(self):
        # --------------------------------------------------------- #
        # Prepare adjoint run
        # --------------------------------------------------------- #
        self.sim.reset_meep()

        self.adjoint_sources = []
        for mi, m in enumerate(self.objective_arguments):
            dJ = grad(self.objective_function,mi)(*self.results_list) # get gradient of objective w.r.t. monitor
            self.adjoint_sources.append(m.place_adjoint_source(dJ)) # place the appropriate adjoint sources

        self.sim.change_sources(self.adjoint_sources)

        # reregsiter design flux
        # FIXME clean up frquency input
        # FIXME cleanup design region input
        # FIXME use yee grid directly 
        self.mon_list = self.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],self.fcen,self.fcen,1,where=self.design_region,yee_grid=False)

        # --------------------------------------------------------- #
        # Adjoint run
        # --------------------------------------------------------- #
        #FIXME make more dynamic
        self.sim.run(until=self.time)

        # --------------------------------------------------------- #
        # Process and store results
        # --------------------------------------------------------- #
        # FIXME allow for multiple design regions
        # FIXME allow for multiple frequencies
        self.a_Ex = self.sim.get_dft_array(self.mon_list,mp.Ex,0)
        self.a_Ey = self.sim.get_dft_array(self.mon_list,mp.Ey,0)
        self.a_Ez = self.sim.get_dft_array(self.mon_list,mp.Ez,0)

    def calculate_gradient(self):
        # FIXME allow for multiple frequencies
        scale = 2 * np.pi * self.fcen * 1j
        # FIXME allow for multiple polarizations/isotropies
        grad = 2 * np.real( (self.d_Ez * self.a_Ez * scale))

        # FIXME allow for multiple design regions
        (x,y,z,w) = self.sim.get_array_metadata(dft_cell=self.mon_list)

        # FIXME allow for multiple dimensions
        x = np.array(x)
        y = np.array(y)

        # FIXME allow for different bases and filters
        self.gradient = self.basis.gradient(grad, x, y)

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
            b0[:] = self.design_function.beta
            # -------------------------------------------- #
            # left function evaluation
            # -------------------------------------------- #
            self.sim.reset_meep()
            
            # assign new design vector
            b0[k] -= db
            self.design_function.set_coefficients(b0)
            
            # initialize design monitors
            for m in self.objective_arguments:
                m.register_monitors()
            
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
            self.design_function.set_coefficients(b0)

            # initialize design monitors
            for m in self.objective_arguments:
                m.register_monitors()
            
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
    
    def update_design(self, beta_vector):
        """Update the design permittivity function.
        """
        self.design_function.set_coefficients(beta_vector)

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