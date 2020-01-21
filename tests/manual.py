import numpy as np
import meep as mp
import matplotlib.pyplot as plt
from scipy import sparse

mp.quiet(quietval=True)

# -------------------------------- #
# Bilinear Interpolation Basis class
# -------------------------------- #

class BilinearInterpolationBasis():
    ''' 
    Simple bilinear interpolation basis set.
    '''
    def __init__(self,region,Nx,Ny):
        ''' '''
        self.region = region
        self.Nx = Nx
        self.Ny = Ny
        self.dim = 2

        # Generate interpolation grid
        self.rho_x = np.linspace(region.center.x - region.size.x/2,region.center.x + region.size.x/2,Nx)
        self.rho_y = np.linspace(region.center.y - region.size.y/2,region.center.y + region.size.y/2,Ny)
    def gradient(self,eps,xtics,ytics):
        # get array of grid points that correspond to epsilon vector
        eps = eps.reshape(eps.size,order='C')
        A = gen_interpolation_matrix(self.rho_x,self.rho_y,xtics,ytics)
        return (eps.T * A).T

    def parameterized_function(self, beta_vector):
        """
        Construct and return a callable, updatable element of the function space.

        After the line
            func = basis.parameterized_function(beta_vector)
        we have:
        1. func is a callable scalar function of one spatial variable:
            func(p) = f_0 + \sum_n beta_n * b_n(p)
        2. func has a set_coefficients method for updating the expansion coefficients
            func.set_coefficients(new_beta_vector)
        """

        class _ParameterizedFunction(object):
            def __init__(self, basis, beta_vector):
                '''
                The user supplied beta vector should be a collapsed column vector of the
                design parameters.
                '''
                self.rho_x = basis.rho_x
                self.rho_y = basis.rho_y
                self.beta  = beta_vector
                
            def set_coefficients(self, beta_vector):
                self.beta = beta_vector
            def __call__(self, p):
                weights, interp_idx = get_bilinear_row(p.x,p.y,self.rho_x,self.rho_y)
                return np.dot( self.beta[interp_idx], weights )
            def func(self):
                def _f(p):
                    return self(p)
                return _f

        return _ParameterizedFunction(self, beta_vector)

# -------------------------------- #
# Helper functions
# -------------------------------- #

def get_bilinear_coefficients(x,x1,x2,y,y1,y2):
    '''
    Calculates the bilinear interpolation coefficients for a single point at (x,y).
    Assumes that the user already knows the four closest points and provides the corresponding
    (x1,x2) and(y1,y2) coordinates.
    '''
    b11 = ((x - x2)*(y - y2))/((x1 - x2)*(y1 - y2))
    b12 = -((x - x2)*(y - y1))/((x1 - x2)*(y1 - y2))
    b21 = -((x - x1)*(y - y2))/((x1 - x2)*(y1 - y2))
    b22 = ((x - x1)*(y - y1))/((x1 - x2)*(y1 - y2))
    return [b11,b12,b21,b22]

def get_bilinear_row(rx,ry,rho_x,rho_y):
    '''
    Calculates a vector of bilinear interpolation weights that can be used
    in an inner product with the neighboring function values, or placed
    inside of an interpolation matrix.
    '''

    Nx = rho_x.size
    Ny = rho_y.size

    # binary search in x direction to get x1 and x2
    xi2 = np.searchsorted(rho_x,rx,side='left')
    if xi2 <= 0: # extrapolation (be careful!)
        xi1 = 0
        xi2 = 1
    elif xi2 >= Nx-1: # extrapolation (be careful!)
        xi1 = Nx-2
        xi2 = Nx-1
    else:
        xi1 = xi2 - 1
    
    x1 = rho_x[xi1]
    x2 = rho_x[xi2]

    # binary search in y direction to get y1 and y2
    yi2 = np.searchsorted(rho_y,ry,side='left')
    if yi2 <= 0: # extrapolation (be careful!)
        yi1 = 0
        yi2 = 1
    elif yi2 >= Ny-1: # extrapolation (be careful!)
        yi1 = Ny-2
        yi2 = Ny-1
    else:
        yi1 = yi2 - 1
    
    y1 = rho_y[yi1]
    y2 = rho_y[yi2]
    
    # get weights
    weights = get_bilinear_coefficients(rx,x1,x2,ry,y1,y2)
    
    # get location of nearest neigbor interpolation points
    interp_idx = np.array([xi1*Nx+yi1,xi1*Nx+yi2,(xi2)*Nx+yi1,(xi2)*Nx+yi2],dtype=np.int64)

    return weights, interp_idx

def gen_interpolation_matrix(rho_x,rho_y,rho_x_interp,rho_y_interp):
    '''
    Generates a bilinear interpolation matrix.
    
    Arguments:
    rho_x ................ [N,] numpy array - original x array mapping to povided data
    rho_y ................ [N,] numpy array - original y array mapping to povided data
    rho_x_interp ......... [N,] numpy array - new x array mapping to desired interpolated data
    rho_y_interp ......... [N,] numpy array - new y array mapping to desired interpolated data

    Returns:
    A .................... [N,M] sparse matrix - interpolation matrix
    '''

    Nx = rho_x.size
    Ny = rho_y.size
    Nx_interp = np.array(rho_x_interp).size
    Ny_interp = np.array(rho_y_interp).size

    input_dimension = Nx * Ny
    output_dimension = Nx_interp * Ny_interp

    interp_weights = np.zeros(4*output_dimension) 
    row_ind = np.zeros(4*output_dimension,dtype=np.int64) 
    col_ind = np.zeros(4*output_dimension,dtype=np.int64)

    ri = 0
    for rx in rho_x_interp:
        for ry in rho_y_interp:
            # get weights
            weights, interp_idx = get_bilinear_row(rx,ry,rho_x,rho_y)

            # populate sparse matrix vectors
            interp_weights[4*ri:4*(ri+1)] = weights
            row_ind[4*ri:4*(ri+1)] = np.array([ri,ri,ri,ri],dtype=np.int64)
            col_ind[4*ri:4*(ri+1)] = interp_idx

            ri += 1
    
    # From matrix vectors, populate the sparse matrix
    A = sparse.coo_matrix((interp_weights, (row_ind, col_ind)),shape=(output_dimension, input_dimension))
    
    return A

#----------------------------------------------------------------------
# Main routine enters here
#----------------------------------------------------------------------
load_from_file = True
for resolution in [40]:

    np.random.seed(64)

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
    forward_sources = [mp.EigenModeSource(mp.GaussianSource(frequency=fcen,fwidth=fwidth),
                        eig_band = 1,
                        direction=mp.NO_DIRECTION,
                        eig_kpoint=kpoint,
                        size = source_size,
                        center=source_center)]
    forward_power = forward_sources[0].eig_power(fcen)

    #----------------------------------------------------------------------
    #- geometric objects
    #----------------------------------------------------------------------
    geometry = [mp.Block(center=mp.Vector3(), material=mp.Medium(index=3.45), size=mp.Vector3(mp.inf, 0.5, 0) )]

    Nx = 4
    Ny = 4

    design_size   = mp.Vector3(1, 1, 0)
    design_region = mp.Volume(center=mp.Vector3(), size=design_size)
    basis = BilinearInterpolationBasis(region=design_region,Nx=Nx,Ny=Ny)

    beta_vector = 11*np.random.rand(Nx*Ny) + 1

    design_function = basis.parameterized_function(beta_vector)
    design_object = [mp.Block(center=design_region.center, size=design_region.size, epsilon_func = design_function.func())]

    geometry += design_object

    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=forward_sources,
                        eps_averaging=False,
                        resolution=resolution)

    #----------------------------------------------------------------------
    #- add monitors
    #----------------------------------------------------------------------

    mon_vol = mp.Volume(center=mp.Vector3(1,0,0),size=mp.Vector3(y=2))
    flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
    #flux = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz],fcen,fcen,1,where=mon_vol)
    design_flux = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],fcen,fcen,1,where=design_region,yee_grid=False)

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
        #tr = np.array(mp.get_fluxes(f_vol))

        # Hm* cross E (c1 term)
        C1 = np.sum(np.conj(Hmy) * Ez - np.conj(Hmz) * Ey,axis=None) * (1/resolution)
        # Em* cross H (c2 term)
        C2 = np.sum(np.conj(Emy) * Hz - np.conj(Emz) * Hy,axis=None) * (1/resolution)
        # Hm* cross Em (first mode overlap)
        MO1 = -np.sum(np.conj(Hmy) * Emz - np.conj(Hmz) * Emy,axis=None) * (1/resolution)
        # Em* cross Hm (second mode overlap)
        MO2 = np.sum(np.conj(Emy) * Hmz - np.conj(Emz) * Hmy,axis=None) * (1/resolution)
        # Normalizer = vgrp * flux_volume(flux)
        normfac = 0.5 * (MO1 + MO2)
        # H* cross E (poynting flux)
        Pin = np.real(np.sum(np.conj(Ey) * Hz - np.conj(Ez) * Hy,axis=None) * (1/resolution))
        # Power flux normalizer
        cscale = np.abs(np.sqrt(1/np.abs(normfac)))
        # Forward coefficient
        cplus = 0.5 * (C2 - C1) * cscale
        # Backward coefficient
        cminus = - 0.5 * (C1 + C2) * cscale

        vgrp = ob.vgrp[0]*2

        ret = np.abs(np.sqrt(1/np.abs(vgrp)))

        print(ret)
        print(vgrp)
        print(cscale)
        #print("cscale: ",cscale)
        #print("vgrp:  ",vgrp)

        #alpha = 0.5 * (C2 / np.conj(Nm) - C1 / Nm)
        #alpha = (C2 - C1) / Pin

        #f = 1/8*np.abs(alpha) ** 2 / Nm
        #A = 1/4*np.conj(alpha) / Nm

        f = np.abs(coeff)**2
        A = ob.alpha[0,0,0]

        
        '''mode_data = EigenmodeData.swigobj 
        print(type(flux))
        mode_mode = np.zeros((2,),dtype=np.complex128)
        #sim.fields.get_mode_mode_overlap(mode_data, mode_data, flux.swigobj, mode_mode)

        print("abs(alec)= {}, angle(alec)={}".format(np.abs(cminus),np.angle(cminus)))
        print("abs(meep)= {}, angle(meep)={}".format(np.abs(coeff),np.angle(coeff)))
        quit()'''
        return f, A, cscale

    f0, alpha, cscale = cost_function(sim,flux)


    # record design cell fields
    d_Ex = sim.get_dft_array(design_flux,mp.Ex,0)
    d_Ey = sim.get_dft_array(design_flux,mp.Ey,0)
    d_Ez = sim.get_dft_array(design_flux,mp.Ez,0)

    #----------------------------------------------------------------------
    #- add adjoint sources
    #----------------------------------------------------------------------

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
    design_flux = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],fcen,fcen,1,where=design_region,yee_grid=False)

    sim.run(until=time)

    #----------------------------------------------------------------------
    #- compute gradient
    #----------------------------------------------------------------------

    envelope = forward_sources[0].src
    freq_env     = envelope.frequency
    factor_env = 1
    if callable(getattr(envelope, "fourier_transform", None)):
        factor_env /= envelope.fourier_transform(freq_env)
    #scale = 1j * np.conj(alpha) / (np.sqrt(adjoint_power) * np.sqrt(forward_power))

    print("forward power: ",forward_power)
    print("factor_env",factor_env)
    print("adjoint_power: ",1/np.sqrt(adjoint_power))
    print("fcen: ",fcen)
    print("alpha: ",np.abs(alpha.conj()))
    print("cscale: ",cscale)
    #quit()
    scale = fcen * 1j * np.conj(alpha)  / resolution / resolution * cscale * 1/np.sqrt(adjoint_power) * 2 * 2 * np.pi
    #scale =  -alpha.conj() / (fcen) / 1j / np.sqrt(adjoint_power) / resolution ** 2
    a_Ex = sim.get_dft_array(design_flux,mp.Ex,0) #* scale 
    a_Ey = sim.get_dft_array(design_flux,mp.Ey,0) #* scale
    a_Ez = sim.get_dft_array(design_flux,mp.Ez,0) #* scale

    (x,y,z,w) = sim.get_array_metadata(dft_cell=design_flux)

    x = np.array(x)
    y = np.array(y)

    #x = (x + 0.5/resolution)[:-1]
    #y = (y + 0.5/resolution)[:-1]


    # Compute dF/deps integral
    grad = np.real( (d_Ez * a_Ez) * scale)

    # Compute dF/rho from dF/deps
    g_adjoint = basis.gradient(grad, x, y)

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


    from os import path
    if path.exists('sweep_{}.npz'.format(resolution)) and load_from_file:
        data = np.load('sweep_{}.npz'.format(resolution))
        idx = data['idx']
        g_discrete = data['g_discrete']
        #g_adjoint = data['g_adjoint']
    
    else:
        g_discrete = 0*np.ones((n,))

        idx = np.random.choice(n,16,replace=False)

        for k in idx:
            
            b0_0 = np.ones((n,))
            b0_1 = np.ones((n,))

            b0_0[:] = beta_vector
            b0_0[k] -= db
            sim.reset_meep()
            design_function.set_coefficients(b0_0)
            flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
            sim.run(until=time)
            fm, _, _ = cost_function(sim,flux)

            b0_1[:] = beta_vector
            b0_1[k] += db
            sim.reset_meep()
            design_function.set_coefficients(b0_1)
            flux = sim.add_flux(fcen,0,1,mp.FluxRegion(center=mon_vol.center,size=mon_vol.size))
            sim.run(until=time)
            fp, _, _ = cost_function(sim,flux)
            
            g_discrete[k] = (fp - fm) / (2*db)
        
        np.savez('sweep_{}.npz'.format(resolution),g_discrete=g_discrete,g_adjoint=g_adjoint,idx=idx)

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
    plt.title('Resolution: {}'.format(resolution))
    plt.legend()
    plt.grid(True)

    plt.savefig('comparison_{}.png'.format(resolution))

plt.show()
