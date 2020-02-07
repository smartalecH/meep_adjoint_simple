import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from . import Basis
from autograd.extend import primitive, defvjp
from autograd.util import func

# -------------------------------- #
# Bilinear Interpolation Basis class
# -------------------------------- #

class BilinearInterpolationBasis(Basis):
    ''' 
    Simple bilinear interpolation basis set.
    '''

    def __init__(self,volume,Nx,Ny,beta):
        ''' 
        
        '''
        self.volume = volume
        self.Nx = Nx
        self.Ny = Ny
        self.dim = 2
        self.num_design_params = self.Nx*self.Ny
        self.beta = beta

        # Generate interpolation grid
        self.rho_x = np.linspace(volume.center.x - volume.size.x/2,volume.center.x + volume.size.x/2,Nx)
        self.rho_y = np.linspace(volume.center.y - volume.size.y/2,volume.center.y + volume.size.y/2,Ny)
    
    @primitive
    def __call__(self, p):
                #weights, interp_idx = self.get_bilinear_row(x,y,self.rho_x,self.rho_y)
                #return np.dot( self.beta[interp_idx], weights )
                if isinstance(p, list):
                    x = np.array([v.x for v in p])
                    y = np.array([v.y for v in p])
                    A = self.gen_interpolation_matrix(self.rho_x,self.rho_y,x,y)
                    return A * self.beta
                else:
                    weights, interp_idx = self.get_bilinear_row(p.x,p.y,self.rho_x,self.rho_y)
                    return np.dot( self.beta[interp_idx], weights )                   
    
    def func(self):
                def _f(p):
                    return self(p)
                return _f

    def gradient(self,p,eps):
        # get array of grid points that correspond to epsilon vector
        #dj_deps = dj_deps.reshape(dj_deps.size,order='C')
        x = [pi.x for pi in p]
        y = [pi.y for pi in p]
        A = self.gen_interpolation_matrix(self.rho_x,self.rho_y,x,y)
        return (eps.T * A).T
        return A

    def set_coefficients(self, beta_vector):
        self.beta = beta_vector
    
    def get_bvector(self, p):
        ''' TODO '''
        return

    def get_bilinear_coefficients(self,x,x1,x2,y,y1,y2):
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

    def get_bilinear_row(self,rx,ry,rho_x,rho_y):
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
        weights = self.get_bilinear_coefficients(rx,x1,x2,ry,y1,y2)
        
        # get location of nearest neigbor interpolation points
        interp_idx = np.array([xi1*Nx+yi1,xi1*Nx+yi2,(xi2)*Nx+yi1,(xi2)*Nx+yi2],dtype=np.int64)

        return weights, interp_idx

    def gen_interpolation_matrix(self,rho_x,rho_y,rho_x_interp,rho_y_interp):
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
                weights, interp_idx = self.get_bilinear_row(rx,ry,rho_x,rho_y)

                # populate sparse matrix vectors
                interp_weights[4*ri:4*(ri+1)] = weights
                row_ind[4*ri:4*(ri+1)] = np.array([ri,ri,ri,ri],dtype=np.int64)
                col_ind[4*ri:4*(ri+1)] = interp_idx

                ri += 1
        
        # From matrix vectors, populate the sparse matrix
        A = sparse.coo_matrix((interp_weights, (row_ind, col_ind)),shape=(output_dimension, input_dimension))
        
        return A

defvjp(func(BilinearInterpolationBasis.__call__), None, lambda ans, f, p, eps: lambda a: f.gradient(p,eps), None)