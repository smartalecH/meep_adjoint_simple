'''
simple_optimization.py
'''

import meep as mp
import meep_adjoint as mpa
import numpy as np
import jax.numpy as npa
from matplotlib import pyplot as plt
from os import path
from scipy import optimize
import nlopt

mp.quiet(quietval=True)
load_from_file = True

#----------------------------------------------------------------------
# Initial setup
#----------------------------------------------------------------------

seed = 24
np.random.seed(seed)
resolution = 10

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx,Sy)

pml_layers = [mp.PML(1.0)]

time = 500

#----------------------------------------------------------------------
# Eigenmode source
#----------------------------------------------------------------------

fcen = 1/1.55
width = 0.2
fwidth = width * fcen
source_center  = [-1.5,0,0]
source_size    = mp.Vector3(0,2,0)
kpoint = mp.Vector3(1,0,0)
src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
source = [mp.EigenModeSource(src,
                    eig_band = 1,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=kpoint,
                    size = source_size,
                    center=source_center)]
#----------------------------------------------------------------------
#- geometric objects
#----------------------------------------------------------------------

Nx = 10
Ny = 10

design_region = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0))
rho_vector = 11*np.random.rand(Nx*Ny) + 1
basis = mpa.BilinearInterpolationBasis(volume=design_region,Nx=Nx,Ny=Ny,rho_vector=rho_vector)

geometry = [
    mp.Block(center=mp.Vector3(x=-Sx/4), material=mp.Medium(index=3.45), size=mp.Vector3(Sx/2, 0.5, 0)), # horizontal waveguide
    mp.Block(center=mp.Vector3(y=Sy/4), material=mp.Medium(index=3.45), size=mp.Vector3(0.5, Sy/2, 0)),  # vertical waveguide
    mp.Block(center=design_region.center, size=design_region.size, epsilon_func=basis.func()) # design region
]

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    eps_averaging=False,
                    resolution=resolution)

#----------------------------------------------------------------------
#- Objective quantities and objective function
#----------------------------------------------------------------------
mode = 1
TE0 = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(x=-1),size=mp.Vector3(y=1.5)),mode)
TE_top = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(0,1,0),size=mp.Vector3(x=1.5)),mode)
ob_list = [TE0,TE_top]

def J(input,top):
    return npa.abs(top/input) ** 2

#----------------------------------------------------------------------
#- Define optimization problem
#----------------------------------------------------------------------
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_function=J,
    objective_arguments=ob_list,
    basis=[basis],
    fcen=fcen,
    time=time
)

#----------------------------------------------------------------------
#- Get gradient funcs
#----------------------------------------------------------------------
def fj(x,grad):
    f, gt = opt(x)
    if grad.size > 0:
        grad[:] = gt
    print("eval: ",min(x),max(x),f)
    return float(f)
algorithm = nlopt.G_MLSL_LDS
local_algorithm = nlopt.LD_SLSQP
opt_alg = nlopt.opt(local_algorithm, int(Nx*Ny))
local_opt = nlopt.opt(local_algorithm, int(Nx*Ny))
local_opt.set_maxeval(12)
opt_alg.set_ftol_rel(1e-4)
#opt_alg.set_local_optimizer(local_opt)
opt_alg.set_max_objective(fj)
opt_alg.set_lower_bounds(1)
opt_alg.set_upper_bounds(12)
opt_alg.set_maxeval(50)
xopt = opt_alg.optimize(rho_vector)

plt.figure()
plt.plot(np.array(opt.f_bank),'o-')

plt.figure()
opt.plot2D()
plt.show()


