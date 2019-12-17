import sys
import os
import argparse

import numpy as np
import meep as mp

import meep_adjoint

from meep_adjoint import get_adjoint_option as adj_opt
from meep_adjoint import get_visualization_option as vis_opt

from meep_adjoint import ( OptimizationProblem, Subregion,
                           ORIGIN, XHAT, YHAT, ZHAT, E_CPTS, H_CPTS, v3, V3)

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mayavi import mlab

######################################################################
# override meep_adjoint's default settings for some configuration opts
######################################################################
meep_adjoint.set_option_defaults( { 'fcen': 0.5, 'df': 0.2,
                                    'dpml': 1.0, 'dair': 1,
                                    'eps_func': 6.0 })


# fetch values of meep_adjoint options that we will use below
fcen = adj_opt('fcen')
dpml = adj_opt('dpml')
dair = adj_opt('dair')


######################################################################
# handle problem-specific command-line arguments
######################################################################
parser = argparse.ArgumentParser()

# options affecting the geometry of the cross-router
parser.add_argument('--width', type=float, default=0.5,  help='width of waveguide')
parser.add_argument('--thickness', type=float, default=0.155,  help='height of waveguide in z-direction')
parser.add_argument('--cladding', type=float, default=1,  help='cladding height on each side of waveguide in z-direction')
parser.add_argument('--l_stub',   type=float, default=3.0,  help='waveguide input/output stub length')
parser.add_argument('--l_design', type=float, default=4.0,  help='design region side length')

# basis-set options
parser.add_argument('--element_length',  type=float,  default=0.25,       help='finite-element length scale')
parser.add_argument('--element_type',    type=str,    default='Lagrange', help='finite-element type')
parser.add_argument('--element_order',   type=int,    default=1,          help='finite-element order')

# configurable weighting prefactors for the north, south, and east power fluxes
# to allow the objective function to be redefined via command-line options
parser.add_argument('--e_weight', type=float, default=1.00, help='')

args = parser.parse_args()


##################################################
# set up optimization problem
##################################################


#----------------------------------------
# size of computational cell
#----------------------------------------
lcen          = 1.0/fcen
dpml          = 0.5*lcen if dpml == -1.0 else dpml
design_length = args.l_design
sx = sy       = dpml + args.l_stub + design_length + args.l_stub + dpml
sz            = 0.0 if args.thickness==0.0 else dpml + dair + args.thickness + dair + dpml
cell_size     = [sx, sy, sz]

#----------------------------------------------------------------------
#- geometric objects (material bodies), not including the design object
#----------------------------------------------------------------------
Si = mp.Medium(epsilon=12.0)
SiO2 = mp.Medium(index=1.44)

cladding = mp.Block(center=V3(ORIGIN),size=cell_size,material=SiO2)
hwvg = mp.Block(center=V3(ORIGIN), material=Si, size=V3(sx, args.width, args.thickness) )
background_geometry = [cladding,hwvg]

#----------------------------------------------------------------------
#- objective regions
#----------------------------------------------------------------------
d_flux     = 0.5*(design_length + args.l_stub)  # distance from origin to NSEW flux monitors
gap        = args.l_stub/6.0                    # gap between source region and flux monitor
d_source   = d_flux + gap                       # distance from origin to source
d_flx2     = d_flux + 2.0*gap

e_center   = ORIGIN + d_flux*XHAT
w1_center  = ORIGIN - d_flux*XHAT
w2_center  = w1_center - 2.0*gap*XHAT

ew_size    = [0.0, 2.0*args.width, sz]

east       = Subregion(center=e_center, size=ew_size, dir=mp.X,  name='east')
west1      = Subregion(center=w1_center, size=ew_size, dir=mp.X, name='west1')
west2      = Subregion(center=w2_center, size=ew_size, dir=mp.X, name='west2')

#----------------------------------------------------------------------
# objective function and extra objective quantities -------------------
#----------------------------------------------------------------------
e_term = '{}*Abs(P1_east)**2'.format(args.e_weight) if args.e_weight else ''

objective = e_term
extra_quantities = ['S_east', 'S_west1', 'S_west2']

#----------------------------------------------------------------------
# source region
#----------------------------------------------------------------------
source_center  = ORIGIN - d_source*XHAT
source_size    = ew_size
source_region  = Subregion(center=source_center, size=source_size, name=mp.X)

#----------------------------------------------------------------------
#- design region, expansion basis
#----------------------------------------------------------------------
design_center = ORIGIN
design_size   = [design_length, design_length, args.thickness]
design_region = Subregion(name='design', center=design_center, size=design_size)

#----------------------------------------
#- optional extra regions for visualization
#----------------------------------------
full_region = Subregion(name='full', center=ORIGIN, size=cell_size)


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
opt_prob = OptimizationProblem(objective_regions=[east, west1, west2],
                               objective_function=objective,
                               design_region=design_region,
                               cell_size=cell_size, 
                               background_geometry=background_geometry,
                               source_region=source_region,
                               extra_quantities=extra_quantities, 
                               extra_regions=[full_region])

f, df = opt_prob.get_fdf_funcs()

    
def J(x,info):
    plt.figure()
    opt_prob.stepper.sim.plot2D()
    filename = 'optiter_{}.png'.format(info['Nfeval'])
    plt.savefig(filename)
    plt.close('all')
    info['Nfeval'] += 1
    return -f(x)

def dJ(x,info):
    return df(x)

global info_iter = 0
def f(x,grad):
    res,grad[:] = opt_prob(x)
    plt.figure()
    opt_prob.stepper.sim.plot2D()
    filename = 'optiter_{}.png'.format(info_iter)
    plt.savefig(filename)
    plt.close('all')
    info_iter += 1
    return res[0]

n = opt_prob.basis.dim
x0 = 3*np.ones((n,))
lb = 1*np.ones((n,))
ub = 12*np.ones((n,))
maxeval = 100
maxtime = 60*5
#res = minimize(J, x0, args=({'Nfeval':0},), method='L-BFGS-B', jac=dJ, options={'maxiter':maxiter, 'disp': True})

algorithm = nlopt.GD STOGO
opt = nlopt.opt(algorithm, n)
opt.set_max_objective(f)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_maxeval(maxeval)
opt.set_maxtime(maxtime)
xopt = opt.optimize(x)
