"""Implementation of TimeStepper class."""

import os
import sys
import time
import numpy as np
import meep as mp

import warnings
from datetime import datetime as dt2

from . import (ObjectiveFunction, Basis, E_CPTS, XHAT, YHAT, ZHAT)

class TimeStepper(object):
    """
    Abstraction of low-level FDTD engine.

    TimeStepper is a class that knows how to invoke an FDTD solver
    to execute a time-domain calculation to obtain frequency-domain
    (DFT) field components in objective regions, from which may be
    computed (a) the values of objective quantities and objective functions,
    or (b) permittivity derivatives of the objective function, from which
    may be computed the objective-function gradient.

    This class lies between the top-level `OptimizationProblem` session-manager class
    and lower-level classes like `ObjectiveFunction`. It exports a single method
    (``__call__``) which prepares and executes one complete FDTD timestepping run to
    compute frequency-domain fields, returning numerical quantities of interest
    computed from these fields. ``__call__`` accepts one `str`-valued argument `job`,
    for which the recognized values are `forward` or `adjoint`. For `job`==`forward`, the
    excitation sources for the FDTD problem are the user-defined sources passed to the
    `OptimizationProblem` constructor and the result of the calculation is the
    objective-function value (plus the values of any additional requested objective quantities).
    For `job`==`adjoint`, the excitation sources are automatically determined internally
    and the result of the calculation is the objective-function gradient.

    `TimeStepper` automatically determines when to terminate a timestepping run by
    monitoring the numerical convergence of the output quantities. The details of this process
    may be customized using configuration options. `TimeStepper` does not itself know how
    to evaluate its output quantities, but rather outsources these calculations to public
    methods of `ObjectiveFunction` and other low-level classes.


    Methods
    -------
    __call__(job):
        Populate the FDTD grid with an appropriate source distribution, initialize DFT cells for
        tabulating frequency-domain fields, then execute FDTD timestepping until the output quantities
        converges and return those quantities.

    """

    #########################################################
    #########################################################
    #########################################################
    def __init__(self, obj_func, dft_cells, basis, sim, fwd_sources):
        """
        Constructor.

        Parameters
        ----------
        obj_func : ObjectiveFunction
            The function whose value or gradient we compute (copied from OptimizationProblem)
        dft_cells : list of `DFTCell`"
            List of `DFTCell` structures, assumed to be ordered with objective cells *first* and the
            design cell *last* (so any `extra` cells are in the interior of the list).
        basis : Basis
            Function space (copied from OptimizationProblem).
            Logically this is not a necessary input for this calculation---it is used only in a
            post-processing step to compute the objective-function gradient by projecting df/dEps
            onto the function space. It might make more sense to define the output of TimeStepper
            to be the raw df/dEps matrix, with the projection step done at a higher level, in which
            case this parameter could be omitted. But then the convergence of the adjoint timestepping
            run would have to consider the full raw df/dEps matrix instead of the lower-dimensional
            gradient vector.
        sim : `meep.Simulation`
            Simulation object, copied from OptimizationProblem.
        fwd_sources : list of `meep.Source`
            User-specified sources for the forward calculation.

        """
        self.obj_func    = obj_func
        self.dft_cells   = dft_cells
        self.design_cell = dft_cells[-1]
        self.basis       = basis
        self.sim         = sim
        self.cmplx       = sim.force_complex_fields
        self.fwd_sources = fwd_sources
        self.dfdEps      = None
        self.state       = 'reset'


    def __update__(self, job):
        """
        Recompute output quantities using most recent values of frequency-domain fields.

        This is an internal helper method for run() that computes
        the objective function value (forward run) or gradient (adjoint run)
        using the latest values of the frequency-domain fields
        stored in the DFT cells. During timestepping it is called
        every check_interval units of MEEP time once the excitation
        sources have died down.

        Parameters
        ----------
        job : str
            'forward' or 'adjoint'


        Returns
        -------
        ** If job==`forward`: **
        fq: numpy array of length N+1 storing values of the objective
            function and the N objective quantities [F, Q0, Q1, ... QN]

        ** If job==`adjoint`: **
        df: numpy array of length `basis.dim` storing components of the
            objective-function gradient

        """
        if job=='forward':
            retvals = self.obj_func(self.dft_cells)
            #print('   ** {:10s}={:.5f}  ** '.format('t',self.sim.round_time()))
            #for n,v in zip(['f'] + self.obj_func.qnames, retvals):
                #print('   ** {:10s}={:+.5e}    '.format(n,v))
        else: # job=='adjoint'
            EH_fwd = self.design_cell.get_EH_slices(label='forward')
            EH_adj = self.design_cell.get_EH_slices()
            self.dfdEps = np.zeros(self.design_cell.grid.shape)
            for n in [ n for (n,c) in enumerate(self.design_cell.components) if c in E_CPTS]:
                self.dfdEps += np.real( EH_fwd[n]*EH_adj[n] )
            vol = 1 / self.sim.resolution ** 2
            self.dfdEps = 2 * vol * self.dfdEps
            retvals = self.basis.gradient(self.dfdEps, self.design_cell.grid)
        return retvals


    #########################################################
    # main timestepper routine that keeps going until the
    # relevant output quantity has converged
    #########################################################
    def run(self, job):
        """Execute a forward or adjoint FDTD timestepping run and return results.

        Parameters
        ----------
        job: str
            'forward' or 'adjoint'

        """
        self.prepare(job)

        last_source_time = self.fwd_sources[0].src.swigobj.last_time()
        max_time         = 10e3#adj_opt('dft_timeout')*last_source_time
        check_interval   = 0.25#adj_opt('dft_interval')*last_source_time
        reltol           = 1.0e-6#adj_opt('dft_reltol')

        # configure real-time animations of evolving time-domain fields
#        step_funcs = []
#        clist = adjoint_adj_opt('animate_components')
#        if clist is not None:
#            ivl=adjoint_adj_opt('animate_interval')
#            ivl=0.5
#            step_funcs = [ AFEClient(self.sim, clist, interval=ivl) ]

        # phase 1: timestepping without interruption until the sources are extinguished
        #prefix = CONSOLE_CODEWORD if adj_opt('silence_meep') else ''
        #log("Beginning {} timestepping run...".format(job))

        #update_dashboard(['run {}'.format(job)])
        #global mt0, wt0, wtdb, wtcpu, proc
        #dummy = proc.cpu_percent()

        #stage, mta, mtb = 'SOURCES', 0, last_source_time
        mtb = last_source_time
        #update_dashboard(['stage {}'.format(stage),
        #                  'progress range {} {}'.format(int(mta),int(mtb))])
        #log('stepping from {} to {}'.format(mta,mtb))
#        sys.stdout.write(prefix + '**stepping from {} to {}\n'.format(mta,mtb))
        #sys.stdout.write(prefix + termsty(job,'1')   + '  '   \
        #                        + termsty(stage,'2') + '  '   \
        #                        + termsty('{:5.1f} -> t -> {:5.1f}'.format(mta,mtb),'3'))
        #mt0, wt0 = mta, time.time()
        #wtdb, wtcpu, dt = wt0, wt0, (mtb-mta)/100.0
        '''if job=='adjoint':
            import matplotlib.pyplot as plt
            self.sim.run(until=20)
            self.sim.plot2D(fields=mp.Ez)
            plt.show()
            quit()'''
        
        self.sim.run(until=600)
        vals = self.__update__(job)

        # now continue timestepping with intermittent convergence checks until
        # we converge or timeout
        '''stage, max_rel_delta = 0, 1.0e9
        while max_rel_delta>reltol and self.sim.round_time() < max_time:
            mta, mtb, stage = mtb, min(mtb + check_interval, max_time), stage+1
            self.sim.run(until=mtb)
            last_vals, vals = vals, self.__update__(job)
            rel_delta = np.array( [rel_diff(v,lv) for v,lv in zip(vals,last_vals)] )
            max_rel_delta = np.amax(rel_delta)
            print('   ** t={} MRD={} ** '.format(self.sim.round_time(), max_rel_delta))'''

        # for forward runs we save the converged DFT fields for later use
        if job=='forward':
            [ cell.save_fields('forward') for cell in self.dft_cells ]
        self.state = job + '.complete'
        return vals


    ##############################################################
    ##############################################################
    ##############################################################
    def prepare(self, job='forward'):
        """ Prepare simulation for timestepping by adding sources and DFT cells.

        Parameters
        ----------
        job (str): The type of timestepping run to prepare:
            1. If job=='forward', prepare forward run to compute objective function value.
            2. If job=='adjoint', prepare adjoint run to compute objective function gradient.
            3. If job is the name of an objective quantity, prepare adjoint run to compute
               the gradient of that quantity.
        """
        target_state = job + '.prepared'
        if self.state == target_state: return

        # get lists of sources and DFT cells to register
        if job=='forward':
            sources = self.fwd_sources
            cells   = self.dft_cells  # for forward runs we must tabulate DFT fields in all DFT cells...
            cmplx   = self.cmplx
        elif job=='adjoint' or job in self.obj_func.qnames:
            sources = self.get_adjoint_sources ( qname = job )
            cells   = [self.design_cell]  # ...for adjoint runs we only need DFT fields in the design region
            cmplx   = True
        else:
            raise ValueError('unknown job {} in TimeStepper.prepare'.format(job))

        # place sources, register cells, initialize fields
        reuse_simulation = False
        self.sim.reset_meep()
        self.sim.change_sources(sources)
        self.force_complex_fields = cmplx
        self.sim.init_sim()
        for cell in cells:
            cell.register(self.sim)
        self.state = target_state


    ##############################################################
    ##############################################################
    ##############################################################
    def get_adjoint_sources(self, qname=None):
        """
        Construct adjoint source distribution.

        Return a list of mp.Source structures describing the distribution of sources
        appropriate for adjoint-method evaluation of the objective-function gradient
        df/deps.

        The optional parameter qname may be set to the name of an objective quantity Q,
        in which case we instead compute dQ/deps.
        """
        ######################################################################
        # extract the temporal envelope of the forward sources and use it to
        # set the overall amplitude prefactor for the adjoint sources
        ######################################################################
        envelope = self.fwd_sources[0].src
        freq     = envelope.frequency
        omega    = freq
        factor   = -1/(1j*omega)
        if callable(getattr(envelope, "fourier_transform", None)):
            factor /= envelope.fourier_transform(freq)

        ######################################################################
        # make a list of (qrule, qweight) pairs for all objective quantities
        # that contribute with nonzero weight to the derivative in question
        ######################################################################
        if qname is None or qname=='adjoint':
            dfdq   = self.obj_func.get_dfdq()
            iwlist = [ (i,w) for i,w in enumerate(dfdq) if w is not 0.0 ]
        elif qname in self.obj_func.qnames:
            iwlist = [ (self.obj_func.qnames.index(qname), 1.0) ]
        else:
            warnings.warn('unknown objective quantity {} in get_adjoint_sources (skipping)'.format(qname))
            return []
        rwlist = [ (self.obj_func.qrules[i], w) for (i,w) in iwlist if w!=0.0 ]

        ######################################################################
        # loop over all contributing objective quantities
        ######################################################################
        sources = []
        for (qrule, qweight) in rwlist:
            code, mode, cell = qrule.code, qrule.mode, self.dft_cells[qrule.ncell]
            EH = cell.get_eigenmode_slices(mode) if mode>0 else cell.get_EH_slices('forward')
            shape = [ len(tics) for tics in [cell.grid.xtics, cell.grid.ytics, cell.grid.ztics] ]

            if code in 'PM':
                sign  =  -1.0 if code=='P' else 1.0
                if cell.normal == 0:
                    kpoint = mp.Vector3(x=1)
                elif cell.normal == 1:
                    kpoint = mp.Vector3(y=1)
                elif cell.normal == 2:
                    kpoint = mp.Vector3(z=1)
                else:
                    raise ValueError("Invalid monitor normal direction")
                
                sources += [mp.EigenModeSource(src=mp.GaussianSource(cell.fcen,fwidth=0.2*cell.fcen),
                                center=cell.region.center,
                                size=cell.region.size,
                                direction=mp.NO_DIRECTION,
                                eig_kpoint=sign*kpoint,
                                amplitude=factor*qweight,
                                #eig_band=bnum,
                                #eig_parity=mp.EVEN_Y+mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
                                eig_match_freq=True
                                )]

                '''sign  =  1.0 if code=='P' else -1.0
                signs = [ +1.0, -1.0, +1.0*sign, -1.0*sign ]
                sources += [ mp.Source(envelope, cell.components[3-nc],
                                       cell.region.center, cell.region.size,
                                       amplitude=signs[nc]*factor*qweight,
                                       amp_data=np.reshape(np.conj(EH[nc]),shape)
                                      ) for nc in range(len(cell.components))
                           ]'''
        return sources


def stop_when_dft_converges():
    return


def rel_diff(a,b):
    """Return value in range [0,2] quantifying error relative to magnitude."""
    diff, scale = np.abs(a-b), np.amax([np.abs(a),np.abs(b)])
    return 2. if np.isinf(scale) else 0. if scale==0. else diff/scale
