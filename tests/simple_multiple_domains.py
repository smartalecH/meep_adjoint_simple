'''
simple.py
'''

import meep as mp
import meep_adjoint as mpa
import autograd.numpy as npa
import numpy as np
from autograd import grad, jacobian, elementwise_grad
from matplotlib import pyplot as plt
from os import path

mp.quiet(quietval=True)
load_from_file = False

#----------------------------------------------------------------------
# Initial setup
#----------------------------------------------------------------------

seed = 24
np.random.seed(seed)
resolution = 20

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
source_center  = [-1,0,0]
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

design_region1 = mp.Volume(center=mp.Vector3(-.5), size=mp.Vector3(0.5, 1, 0))
rho_vector1 = 11*np.random.rand(Nx*Ny) + 1
basis1 = mpa.BilinearInterpolationBasis(volume=design_region1,Nx=Nx,Ny=Ny,rho_vector=rho_vector1)

design_region2 = mp.Volume(center=mp.Vector3(.5), size=mp.Vector3(0.5, 1, 0))
rho_vector2 = 11*np.random.rand(Nx*Ny) + 1
basis2 = mpa.BilinearInterpolationBasis(volume=design_region2,Nx=Nx,Ny=Ny,rho_vector=rho_vector2)

bases = [basis1,basis2]
geometry = [
    mp.Block(center=mp.Vector3(x=-Sx/4), material=mp.Medium(index=3.45), size=mp.Vector3(Sx/2, 0.5, 0)), # horizontal waveguide
    mp.Block(center=mp.Vector3(y=Sy/4), material=mp.Medium(index=3.45), size=mp.Vector3(0.5, Sy/2, 0)),  # vertical waveguide
    mp.Block(center=design_region1.center, size=design_region1.size, epsilon_func=basis1.func()), # design region1,
    mp.Block(center=design_region2.center, size=design_region2.size, epsilon_func=basis2.func())  # design region2
]

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    eps_averaging=False,
                    resolution=resolution)

'''sim.plot2D()
plt.show()
quit()'''
#----------------------------------------------------------------------
#- Objective quantities and objective function
#----------------------------------------------------------------------

TE0 = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(0,1,0),size=mp.Vector3(x=2)),mode=1)
ob_list = [TE0]

def J(alpha):
    return npa.sum(npa.abs(alpha) ** 2)

#----------------------------------------------------------------------
#- Define optimization problem
#----------------------------------------------------------------------
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_function=J,
    objective_arguments=ob_list,
    basis=bases,
    fcen=fcen,
    time=time
)

#----------------------------------------------------------------------
#- Get gradient
#----------------------------------------------------------------------

f0, g_adjoint = opt()

#----------------------------------------------------------------------
#- FD run
#----------------------------------------------------------------------
db = 1e-3
n = Nx*Ny
choose = 20
if mp.am_master():
    if path.exists('simple_multiple_domains_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny)) and load_from_file:
        data = np.load('simple_multiple_domains_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny))
        idx = data['idx']
        g_discrete = data['g_discrete']

    else:
        g_discrete, idx = opt.calculate_fd_gradient(num_gradients=choose,db=db)

    print("Chosen indices: ",idx)
    print("adjoint method: {}".format(g_adjoint[0][idx]))
    print("discrete method: {}".format(g_discrete[idx]))
    print("ratio: {}".format(g_adjoint[0][idx]/g_discrete[idx]))

    (m, b) = np.polyfit(g_discrete, g_adjoint[0], 1)
    print("slope: {}".format(m))

    min = np.min(g_discrete)
    max = np.max(g_discrete)

    plt.figure()
    plt.plot([min, max],[min, max],label='y=x comparison')
    plt.plot(g_discrete[idx],g_adjoint[0][idx],'o',label='Adjoint comparison')
    plt.xlabel('Finite Difference Gradient')
    plt.ylabel('Adjoint Gradient')
    plt.title('Resolution: {} Seed: {} Nx: {} Ny: {}'.format(resolution,seed,Nx,Ny))
    plt.legend()
    plt.grid(True)

    np.savez('simple_multiple_domains_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny),g_discrete=g_discrete,g_adjoint=g_adjoint[0],idx=idx,m=m,b=b,resolution=resolution)
    plt.savefig('simple_multiple_domains_{}_seed_{}_Nx_{}_Ny_{}.png'.format(resolution,seed,Nx,Ny))

    plt.show()