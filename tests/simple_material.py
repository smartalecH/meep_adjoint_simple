'''
simple_material.py
'''

import meep as mp
import meep_adjoint as mpa
import numpy as np
import jax.numpy as npa
from matplotlib import pyplot as plt
from os import path

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

design_region = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0))
rho_vector = 11*np.random.rand(Nx*Ny) + 1

def material_map(sigma):
    sigma_Si = sigma
    cSi_range = mp.FreqRange(min=um_scale, max=um_scale/0.4)

    cSi_frq1 = 3.64/um_scale
    cSi_gam1 = 0
    cSi_sig1 = 8
    cSi_frq2 = 2.76/um_scale
    cSi_gam2 = 2*0.063/um_scale
    cSi_sig2 = 2.85
    cSi_frq3 = 1.73/um_scale
    cSi_gam3 = 2*2.5/um_scale
    cSi_sig3 = -0.107

    cSi_susc = [mp.LorentzianSusceptibility(frequency=cSi_frq1, gamma=cSi_gam1, sigma=cSi_sig1*sigma_Si),
                mp.LorentzianSusceptibility(frequency=cSi_frq2, gamma=cSi_gam2, sigma=cSi_sig2*sigma_Si),
                mp.LorentzianSusceptibility(frequency=cSi_frq3, gamma=cSi_gam3, sigma=cSi_sig3*sigma_Si)]

    cSi = mp.Medium(epsilon=1.0, E_susceptibilities=cSi_susc, valid_freq_range=cSi_range)
    return cSi
sample_mat = material_map
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
    basis=[basis],
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
    if path.exists('simple_material_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny)) and load_from_file:
        data = np.load('simple_material_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny))
        idx = data['idx']
        g_discrete = data['g_discrete']

    else:
        g_discrete, idx = opt.calculate_fd_gradient(num_gradients=choose,db=db)

    print("Chosen indices: ",idx)
    print("adjoint method: {}".format(g_adjoint[idx]))
    print("discrete method: {}".format(g_discrete[idx]))
    print("ratio: {}".format(g_adjoint[idx]/g_discrete[idx]))

    (m, b) = np.polyfit(g_discrete, g_adjoint, 1)
    print("slope: {}".format(m))

    min = np.min(g_discrete)
    max = np.max(g_discrete)

    plt.figure()
    plt.plot([min, max],[min, max],label='y=x comparison')
    plt.plot(g_discrete[idx],g_adjoint[idx],'o',label='Adjoint comparison')
    plt.xlabel('Finite Difference Gradient')
    plt.ylabel('Adjoint Gradient')
    plt.title('Resolution: {} Seed: {} Nx: {} Ny: {}'.format(resolution,seed,Nx,Ny))
    plt.legend()
    plt.grid(True)

    np.savez('simple_material_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny),g_discrete=g_discrete,g_adjoint=g_adjoint,idx=idx,m=m,b=b,resolution=resolution)
    plt.savefig('simple_material_{}_seed_{}_Nx_{}_Ny_{}.png'.format(resolution,seed,Nx,Ny))

    plt.show()