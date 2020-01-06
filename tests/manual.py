import numpy as np
import meep as mp
import meep_adjoint as mpa
import matplotlib.pyplot as plt

mp.quiet(quietval=True)
np.random.seed(64)

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx,Sy)

pml_layers = [mp.PML(1.0)]

resolution = 10

time = 1200

#----------------------------------------------------------------------
# Eigenmode source
#----------------------------------------------------------------------
fcen = 1/1.55
fwidth = 0.1 * fcen
source_center  = [-1,0,0]
source_size    = mp.Vector3(0,2,0)
kpoint = mp.Vector3(1,0,0)
sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                     eig_band = 1,
                     direction=mp.NO_DIRECTION,
                     eig_kpoint=kpoint,
                     size = source_size,
                     center=source_center)]
forward_power = sources[0].eig_power(fcen)
#----------------------------------------------------------------------
#- geometric objects
#----------------------------------------------------------------------
geometry = [mp.Block(center=mp.Vector3(), material=mp.Medium(index=3.45), size=mp.Vector3(mp.inf, 0.5, 0) )]

Nx = 6
Ny = 6
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
#- add monitors
#----------------------------------------------------------------------

mon_vol = mp.Volume(center=mp.Vector3(1,0,0),size=mp.Vector3(y=2))
flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
design_flux = sim.add_flux(fcen,0,1,mp.FluxRegion(direction=mp.X,center=design_region.center,size=design_region.size))

#----------------------------------------------------------------------
#- run forward simulation
#----------------------------------------------------------------------

sim.run(until=time)

#----------------------------------------------------------------------
#- objective function
#----------------------------------------------------------------------

# pull eigenmode data
def cost_function(sim,f_vol):
    mode = 1
    EigenmodeData = sim.get_eigenmode(fcen, mp.X, f_vol.where, mode, mp.Vector3())
    x,y,z,w = sim.get_array_metadata(dft_cell=f_vol)
    x = np.array(x)
    y = np.array(y)

    Emy = np.zeros((x.size,y.size),dtype=np.complex128)
    Emz = np.zeros((x.size,y.size),dtype=np.complex128)
    Hmy = np.zeros((x.size,y.size),dtype=np.complex128)
    Hmz = np.zeros((x.size,y.size),dtype=np.complex128)

    Ey = np.zeros((x.size,y.size),dtype=np.complex128)
    Ez = np.zeros((x.size,y.size),dtype=np.complex128)
    Hy = np.zeros((x.size,y.size),dtype=np.complex128)
    Hz = np.zeros((x.size,y.size),dtype=np.complex128)

    mcomps = [Emy,Emz,Hmy,Hmz]
    sim_comps = [Ey,Ez,Hy,Hz]

    for ic,c in enumerate([mp.Ey,mp.Ez,mp.Hy,mp.Ez]):
        sim_comps[ic][:] = sim.get_dft_array(flux,c,0)
        for ix,px in enumerate(x):
            for iy,py in enumerate(y):
                mcomps[ic][ix,iy] = EigenmodeData.amplitude(mp.Vector3(px,py),c)
    
    ob = sim.get_eigenmode_coefficients(f_vol,[1])
    coeff = ob.alpha[0,0,0]

    # Hm* cross E
    C1 = np.sum(np.conj(Hmy) * Ez - np.conj(Hmz) * Ey,axis=None) * (1/resolution) ** 2
    # Em* cross H
    C2 = np.sum(np.conj(Emy) * Hz - np.conj(Emz) * Hy,axis=None) * (1/resolution) ** 2
    # Hm* cross Em
    Nm = -np.sum(np.conj(Hmy) * Emz - np.conj(Hmz) * Emy,axis=None) * (1/resolution) ** 2
    # H* cross E
    Pin = -np.sum(np.conj(Hy) * Ez - np.conj(Hz) * Ey,axis=None) * (1/resolution) ** 2

    #alpha = 0.5 * (C2 / np.conj(Nm) - C1 / Nm)
    alpha = C2 - C1

    f = np.abs(coeff)**2 #1/8*np.abs(alpha) ** 2 / Nm

    A = coeff#1/4*np.conj(alpha) / Nm

    print(A)

    return f, A

f0, alpha = cost_function(sim,flux)


# record design cell fields
d_Ex = sim.get_dft_array(design_flux,mp.Ex,0)
d_Ey = sim.get_dft_array(design_flux,mp.Ey,0)
d_Ez = sim.get_dft_array(design_flux,mp.Ez,0)

#----------------------------------------------------------------------
#- add adjoint sources
#----------------------------------------------------------------------
#scalar = -2 / (1j*fcen*np.abs(Nm)**2) * 1/resolution * np.conjugate(alpha)

sim.reset_meep()

# update sources
kpoint = mp.Vector3(-1,0,0)
sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                     eig_band = 1,
                     direction=mp.NO_DIRECTION,
                     eig_kpoint=kpoint,
                     size = mon_vol.size,
                     center=mon_vol.center)]
sim.change_sources(sources)
adjoint_power = sources[0].eig_power(fcen)
#----------------------------------------------------------------------
#- run adjoint simulation
#----------------------------------------------------------------------
design_flux = sim.add_flux(fcen,0,1,mp.FluxRegion(direction=mp.X,center=design_region.center,size=design_region.size))

sim.run(until=time)

#----------------------------------------------------------------------
#- compute gradient
#----------------------------------------------------------------------
#scale = 1j * np.conj(alpha) / (np.sqrt(adjoint_power) * np.sqrt(forward_power))

#scale = np.conj(alpha / forward_power) * fcen * 1j / np.sqrt(adjoint_power)
scale = alpha.conj() * fcen * 1j / np.sqrt(adjoint_power)
a_Ex = sim.get_dft_array(design_flux,mp.Ex,0) #* scale 
a_Ey = sim.get_dft_array(design_flux,mp.Ey,0) #* scale
a_Ez = sim.get_dft_array(design_flux,mp.Ez,0) #* scale

'''plt.figure()
plt.imshow(np.real(a_Ez))

plt.figure()
plt.imshow(np.real(d_Ez))

plt.show()
quit()'''

(x,y,z,w) = sim.get_array_metadata(dft_cell=design_flux)
#x = x[1:-1]
#y = y[1:-1]

grid = mpa.xyzw2grid((x,y,z,w))

grad = np.real(2 * (d_Ex * a_Ex + d_Ey * a_Ey + d_Ez * a_Ez) * scale * (1/resolution ** 2))
#grad = grad[1:-1,1:-1]
g_adjoint = basis.gradient(grad, grid)

'''plt.figure()
plt.imshow(grad)

plt.figure()
print(grad)
plt.imshow(g_adjoint.reshape((Nx,Ny),order='C'))

plt.show()
quit()'''

sim.reset_meep()
kpoint = mp.Vector3(1,0,0)
sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                     eig_band = 1,
                     direction=mp.NO_DIRECTION,
                     eig_kpoint=kpoint,
                     size = source_size,
                     center=source_center)]

sim.change_sources(sources)
#----------------------------------------------------------------------
#- compute finite difference approximate
#----------------------------------------------------------------------

db = 1e-4
n = Nx*Ny
g_discrete = 0*np.ones((n,))

idx = np.random.randint(0,Nx*Ny,100)

for k in idx:
    
    b0_0 = np.ones((n,))
    b0_1 = np.ones((n,))

    b0_0[:] = beta_vector
    b0_0[k] -= db
    sim.reset_meep()
    design_function.set_coefficients(b0_0)
    flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
    sim.run(until=time)
    fm, _ = cost_function(sim,flux)

    b0_1[:] = beta_vector
    b0_1[k] += db
    sim.reset_meep()
    design_function.set_coefficients(b0_1)
    flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
    sim.run(until=time)
    fp, _ = cost_function(sim,flux)
    
    g_discrete[k] = (fp - fm) / (2*db)

print("Chosen indices: ",idx)
print("adjoint method: {}".format(g_adjoint[idx]))
print("discrete method: {}".format(g_discrete[idx]))
print("ratio: {}".format(g_adjoint[idx]/g_discrete[idx]))


plt.figure()
plt.plot(g_discrete[idx],g_adjoint[idx],'o')

plt.show()
