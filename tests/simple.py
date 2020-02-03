import meep as mp
import meep_adjoint as mpa
import autograd.numpy as npa
import numpy as np
from autograd import grad
from matplotlib import pyplot as plt

mp.quiet(quietval=True)
load_from_file = True
#----------------------------------------------------------------------
# Initial setup
#----------------------------------------------------------------------

seed = 24
np.random.seed(seed)

resolution = 50

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx,Sy)

pml_layers = [mp.PML(1.0)]

time = 1200

#----------------------------------------------------------------------
# Eigenmode source
#----------------------------------------------------------------------

fcen = 1/1.55
width = 0.1
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

geometry = [mp.Block(center=mp.Vector3(), material=mp.Medium(index=3.45), size=mp.Vector3(mp.inf, 0.5, 0) )]

Nx = 3
Ny = 3

design_size   = mp.Vector3(1, 1, 0)
design_region = mp.Volume(center=mp.Vector3(), size=design_size)
basis = mpa.BilinearInterpolationBasis(region=design_region,Nx=Nx,Ny=Ny)

beta_vector = 11*np.random.rand(Nx*Ny) + 1

design_function = basis.parameterized_function(beta_vector)
design_object = [mp.Block(center=design_region.center, size=design_region.size, epsilon_func = design_function.func())]

geometry += design_object

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    eps_averaging=False,
                    resolution=resolution)

#----------------------------------------------------------------------
#- Objective quantities and objective function
#----------------------------------------------------------------------

EMC_vol = mp.Volume(center=mp.Vector3(1,0,0),size=mp.Vector3(y=2))
TE0 = mpa.EigenmodeCoefficient(sim,EMC_vol,fcen,0,1,1,src)
ob_list = [TE0]

def J(alpha):
    return npa.abs(alpha) ** 2

#----------------------------------------------------------------------
#- Define optimization problem
#----------------------------------------------------------------------

opt = mpa.OptimizationProblem(
    simulation = sim,
    objective_function = J,
    objective_arguments = ob_list,
    design_function = design_function,
    basis = basis,
    fcen = fcen
)

#----------------------------------------------------------------------
#- Get gradient
#----------------------------------------------------------------------

f0, g_adjoint = opt()

#----------------------------------------------------------------------
#- FD run
#----------------------------------------------------------------------
sim.reset_meep()
kpoint = mp.Vector3(1,0,0)
sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                    eig_band = 1,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=kpoint,
                    size = source_size,
                    center=source_center)]

sim.change_sources(sources)

db = 1e-4
n = Nx*Ny
choose = n

from os import path
if path.exists('sweep_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny)) and load_from_file:
    data = np.load('sweep_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny))
    idx = data['idx']
    g_discrete = data['g_discrete']
    #g_adjoint = data['g_adjoint']

else:
    g_discrete = 0*np.ones((n,))

    idx = np.random.choice(n,choose,replace=False)

    for k in idx:
        
        b0_0 = np.ones((n,))
        b0_1 = np.ones((n,))

        b0_0[:] = beta_vector
        b0_0[k] -= db
        sim.reset_meep()
        design_function.set_coefficients(b0_0)
        EMC_vol = mp.Volume(center=mp.Vector3(1,0,0),size=mp.Vector3(y=2))
        TE0 = mpa.EigenmodeCoefficient(sim,EMC_vol,fcen,0,1,1,src)
        TE0.register_monitors()
        sim.run(until=time)
        alpha = TE0()
        fm = J(alpha)

        b0_1[:] = beta_vector
        b0_1[k] += db
        sim.reset_meep()
        design_function.set_coefficients(b0_1)
        EMC_vol = mp.Volume(center=mp.Vector3(1,0,0),size=mp.Vector3(y=2))
        TE0 = mpa.EigenmodeCoefficient(sim,EMC_vol,fcen,0,1,1,src)
        TE0.register_monitors()
        sim.run(until=time)
        alpha = TE0()
        fp = J(alpha)
        
        g_discrete[k] = (fp - fm) / (2*db)

#----------------------------------------------------------------------
#- Compare
#----------------------------------------------------------------------

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


np.savez('sweep_{}_seed_{}_Nx_{}_Ny_{}.npz'.format(resolution,seed,Nx,Ny),g_discrete=g_discrete,g_adjoint=g_adjoint,idx=idx,m=m,b=b,resolution=resolution)
plt.savefig('comparison_{}_seed_{}_Nx_{}_Ny_{}.png'.format(resolution,seed,Nx,Ny))

plt.show()