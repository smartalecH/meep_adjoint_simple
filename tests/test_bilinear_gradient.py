import numpy as np
import meep as mp
import meep_adjoint as mpa
import matplotlib.pyplot as plt
from autograd import numpy as npa
from autograd import grad

mp.quiet(quietval=True)
np.random.seed(64)

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx,Sy)

pml_layers = [mp.PML(1.0)]

resolution = 20

#----------------------------------------------------------------------
# Eigenmode source
#----------------------------------------------------------------------
fcen = 1/1.55
fwidth = 0.1 * fcen
source_center  = [-1.5,0,0]
source_size    = 2.0*mpa.YHAT
kpoint = 3*mpa.XHAT
sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                     eig_band = 1,
                     direction=mp.NO_DIRECTION,
                     eig_kpoint=kpoint,
                     size = source_size,
                     center=source_center)]

#----------------------------------------------------------------------
#- geometric objects
#----------------------------------------------------------------------
geometry = [mp.Block(center=mp.Vector3(), material=mp.Medium(index=3.45), size=mp.Vector3(mp.inf, 0.5, 0) )]

Nx = 10
Ny = 10
design_size   = mp.Vector3(1, 1, 0)
design_region = mpa.Subregion(name='design', center=mp.Vector3(), size=design_size)
basis = mpa.BilinearInterpolationBasis(region=design_region,Nx=Nx,Ny=Ny)

beta_vector = 11*np.random.rand(Nx*Ny) + 1

design_function = basis.parameterized_function(beta_vector)
design_object = [mp.Block(center=design_region.center, size=design_region.size, epsilon_func = design_function.func())]

geometry += design_object

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    eps_averaging=False,
                    resolution=resolution)

#----------------------------------------------------------------------
#- objective regions
#----------------------------------------------------------------------

e_center   = 1*mpa.XHAT
w1_center  = -1*mpa.XHAT
w2_center  = -2*mpa.XHAT

east       = mpa.Subregion(center=e_center,  size=2.0*mpa.YHAT,  dir=mp.X,  name='east')
west1      = mpa.Subregion(center=w1_center, size=2.0*mpa.YHAT,  dir=mp.X,  name='west1')
west2      = mpa.Subregion(center=w2_center, size=2.0*mpa.YHAT,  dir=mp.X,  name='west2')

objective_regions = [east, west1, west2]

#----------------------------------------------------------------------
# objective function and extra objective quantities -------------------
#----------------------------------------------------------------------
splitter = False
fobj_router   = '|P1_north/P1_west1|^2'
fobj_router = '|P1_east|^2'
fobj_splitter = '( |P1_north| - |P1_east| )^2 + ( |P1_east| - |M1_south| )^2'
objective_function = fobj_router
extra_quantities = ['S_north', 'S_south', 'S_east', 'S_west1', 'S_west2']

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

opt_prob = mpa.OptimizationProblem(
    sim,
    objective_regions=objective_regions,
    objective_function=objective_function,
    basis=basis,
    beta_vector=beta_vector,
    design_function=design_function
)
f_adjoint, g_adjoint = opt_prob(beta_vector)

#----------------------------------------------------------------------
# -- Solve discrete problem
#----------------------------------------------------------------------
db = 1e-5
n = Nx*Ny
g_discrete = 0*np.ones((n,))

idx = np.random.rand(4)*Nx*Ny
idx = idx.astype(np.int64)

for k in idx:
    b0_0 = np.ones((n,))
    b0_1 = np.ones((n,))

    b0_0[:] = beta_vector
    b0_0[k] -= db
    temp, _ = opt_prob(b0_0,need_gradient=False)
    f0 = np.real(temp[0])

    b0_1[:] = beta_vector
    b0_1[k] += db
    temp, _ = opt_prob(b0_1,need_gradient=False)
    f1 = np.real(temp[0])
    
    g_discrete[k] = (f1 - f0) / (2*db)
#----------------------------------------------------------------------
# -- Compare
#----------------------------------------------------------------------

print("Chosen indices: ",idx)
print("adjoint method: {}".format(g_adjoint[idx]))
print("discrete method: {}".format(g_discrete[idx]))
print("ratio: {}".format(g_adjoint[idx]/g_discrete[idx]))