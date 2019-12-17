import meep as mp
import meep_adjoint as mpa
import numpy as np
from matplotlib import pyplot as plt
import nlopt

mp.quiet(quietval=True)

#----------------------------------------
# Simulation wavelength specs
#----------------------------------------
wavelength = 1.55
fcen = 1/wavelength
nfreq = 1
df = 0
fwidth = 0.1*fcen

#----------------------------------------
# Simulation wavelength specs
#----------------------------------------
waveguide_width = 0.5
waveguide_h     = 0
l_stub          = 3.0                         # waveguide stub length
l_design        = 4.0                         # design region side length

#----------------------------------------
# computational cell
#----------------------------------------
# PML regions
lcen          = 1.0/fcen
dpml          = 0.5*lcen
pml_layers = [mp.PML(dpml)]

# Simulation domain boundaries
sx = sy       = dpml + l_stub + l_design + l_stub + dpml
sz            = 0.0 if waveguide_h==0.0 else (dpml + dair + waveguide_h + dair + dpml)

# Monitor locations
d_flux        = 0.5*(l_design + l_stub)     # distance from origin to NSEW flux monitors
d_flx2        = d_flux + l_stub/3.0         # distance from origin to west2 flux monitor
d_source      = d_flux + l_stub/6.0         # distance from origin to source
cell_size     = [sx, sy, sz]

resolution = 10

#----------------------------------------------------------------------
#- geometric objects (material bodies), not including the design object
#----------------------------------------------------------------------
Si   = mp.Medium(index=3.45)
SiO2 = mp.Medium(index=1.44)

waveguide_width = 0.5

wvg_horizontal   = mp.Block(center=mp.Vector3(), material=Si, size=mp.Vector3(mp.inf, waveguide_width,  waveguide_h) )
wvg_vertical     = mp.Block(center=mp.Vector3(), material=Si, size=mp.Vector3(waveguide_width, mp.inf, waveguide_h) )
south_wvg  = mp.Block(center=mp.Vector3(-0.25*sy*mpa.YHAT), material=Si, size=mp.Vector3(waveguide_width, 0.5*sy, waveguide_h) )

geometry = [ wvg_horizontal, wvg_vertical]

#----------------------------------------------------------------------
# Eigenmode source
#----------------------------------------------------------------------
source_center  = - d_source*mpa.XHAT
source_size    = 2.0*waveguide_width*mpa.YHAT
sources = [mp.Source(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                     component=mp.Ez,
                     size = source_size,
                     center=source_center)]
#----------------------------------------------------------------------
#- design region and basis
#----------------------------------------------------------------------

design_size   = mp.Vector3(l_design, l_design, waveguide_h)
design_region = mpa.Subregion(fcen, df, nfreq, name='design', center=mp.Vector3(), size=design_size)
element_length = 0.2
element_type = 'CG 1'

basis = mpa.FiniteElementBasis(region=design_region,
                               element_length=element_length,
                               element_type=element_type)

beta_vector = basis.project(3.45**2)
design_function = basis.parameterized_function(beta_vector)
design_object = [mp.Block(center=design_region.center, size=design_region.size, epsilon_func = design_function.func())]

geometry += design_object

#----------------------------------------------------------------------
#- set up simulation object
#----------------------------------------------------------------------

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#----------------------------------------------------------------------
#- objective regions
#----------------------------------------------------------------------
n_center   = d_flux*mpa.YHAT
s_center   = -d_flux*mpa.YHAT
e_center   = d_flux*mpa.XHAT
w1_center  = -d_flux*mpa.XHAT
w2_center  = w1_center - (l_stub/3.0)*mpa.XHAT

north      = mpa.Subregion(fcen, df, nfreq, center=n_center,  size=2.0*waveguide_width*mpa.XHAT, dir=mp.Y,  name='north')
south      = mpa.Subregion(fcen, df, nfreq, center=s_center,  size=2.0*waveguide_width*mpa.XHAT, dir=mp.Y,  name='south')
east       = mpa.Subregion(fcen, df, nfreq, center=e_center,  size=2.0*waveguide_width*mpa.YHAT,  dir=mp.X,  name='east')
west1      = mpa.Subregion(fcen, df, nfreq, center=w1_center, size=2.0*waveguide_width*mpa.YHAT,  dir=mp.X,  name='west1')
west2      = mpa.Subregion(fcen, df, nfreq, center=w2_center, size=2.0*waveguide_width*mpa.YHAT,  dir=mp.X,  name='west2')

objective_regions = [north, south, east, west1, west2]

#----------------------------------------------------------------------
# objective function and extra objective quantities -------------------
#----------------------------------------------------------------------
splitter = False
fobj_router   = '|P1_north/P1_west1|^2'
fobj_splitter = '( |P1_north| - |P1_east| )^2 + ( |P1_east| - |M1_south| )^2'
objective_function = fobj_splitter if splitter else fobj_router
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

it = 0
results = []
def J(b,grad):
    global it
    f, g = opt_prob(b)
    grad[:] = np.real(g)
    results.append(f[0])
    opt_prob.visualize()
    #plt.imshow(np.rot90(fields),alpha=0.3)
    plt.savefig('state_{}.png'.format(it))
    it+=1
    return np.real(f[0])

n = basis.dim
b0 = 3.45**2*np.random.rand(n) + 1
lb = np.ones((n,))
ub = 3.45**2*np.ones((n,)) + 1
maxeval = 50
maxtime = 10*60

algorithm = nlopt.LD_TNEWTON
opt = nlopt.opt(algorithm, n)
opt.set_max_objective(J)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_maxeval(maxeval)
opt.set_maxtime(maxtime)
bopt = opt.optimize(b0)
print(bopt)

plt.figure()
opt_prob.visualize()

plt.figure()
plt.plot(results,'-o')

plt.show()

