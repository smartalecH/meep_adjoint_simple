import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from . import Basis

# -------------------------------- #
# Bilinear Interpolation Basis class
# -------------------------------- #

class BilinearInterpolationBasis(Basis):
    ''' 
    Simple bilinear interpolation basis set.
    '''

    def __init__(self,region,Nx,Ny):
        ''' '''
        self.region = region
        self.Nx = Nx
        self.Ny = Ny
        self.dim = 2
        self.num_design_params = self.Nx*self.Ny

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

    
    def get_bvector(self, p):
        ''' TODO '''
        return

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

