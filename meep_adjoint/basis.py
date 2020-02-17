import numpy as np
import meep as mp
from autograd import numpy as npa
from scipy import sparse
from autograd import grad, jacobian, vector_jacobian_product
#invoke python's 'abstract base class' formalism in a version-agnostic way
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {'__slots__': ()}) # compatible with Python 2 and 3

#----------------------------------------------------------------------
# Basis is the abstract base class from which classes describing specific
# basis sets should inherit.
#----------------------------------------------------------------------
class Basis(ABC):
    """
    """

    def __init__(self, rho_vector, volume=None, size=None, center=mp.Vector3(), filter=None, material_mapping=None):
        self.volume = volume if volume else mp.Volume(center=center,size=size)
        self.filter = filter
        self.rho_vector = rho_vector
        self.rho_prime_vector = rho_vector if self.filter is None else self.filter(rho_vector)

        #TODO implment material_mapping
        self.material_mapping=material_mapping
    
    def gradient(self,d_E,a_E,frequency_scalar,design_grid):
        '''
        '''

        # Chain rule for the material_mapping
        if self.material_mapping is None:
            # Propogate out the frequencies and components. Assume no dispersion, and assume isotropy.
            # FIXME use tensordot instead...
            # gradient = np.tensordot(d_E,a_E,axes=([1,0],[0,1]))
            N = frequency_scalar.size
            dJ_deps = np.zeros((len(design_grid.x),len(design_grid.y),len(design_grid.z)))
            for ix in range(len(design_grid.x)):
                for iy in range(len(design_grid.y)):
                    for iz in range(len(design_grid.z)):
                        for ic in [2]:
                            dJ_deps[ix,iy,iz] += 2 * np.sum(np.real(frequency_scalar * a_E[ix,iy,iz,ic,:] * d_E[ix,iy,iz,ic,:]))
        else:
            raise NotImplementedError("Material maps are not yet implemented")
        
        # Chain rule for the basis interpolator
        dJ_deps = dJ_deps.reshape(dJ_deps.size,order='C')
        dJ_dp = dJ_deps * self.get_basis_jacobian(design_grid)
        
        # Chain rule for the filtering functions
        # FIXME cleanup when no filter
        if self.filter is None:
            dJ_drho = dJ_dp
        else:
            dJ_drho = np.matmul(jacobian(self.filter)(dJ_dp), dJ_dp)
        return dJ_drho
    
    def func(self):
        def _f(p): 
            return self(p)
        return _f

    @abstractmethod
    def get_basis_jacobian(self):
        raise NotImplementedError("derived class must implement __call__() method")
    
    @abstractmethod
    def __call__(self, p=[0.0,0.0]):
        raise NotImplementedError("derived class must implement __call__() method")

    def set_rho_vector(self, rho_vector):
        self.rho_vector = rho_vector
        self.rho_prime_vector = rho_vector if self.filter is None else self.filter(rho_vector)

# -------------------------------- #
# Bilinear Interpolation Basis class
# -------------------------------- #

class BilinearInterpolationBasis(Basis):
    ''' 
    Simple bilinear interpolation basis set.
    '''

    def __init__(self,Nx,Ny,**kwargs):
        ''' 
        
        '''
        self.Nx = Nx
        self.Ny = Ny
        # FIXME allow for 3d geometries using 2d bilinear interpolation
        self.dim = 2
        self.num_design_params = self.Nx*self.Ny

        super(BilinearInterpolationBasis, self).__init__(**kwargs)

        # Generate interpolation grid
        self.rho_x = np.linspace(self.volume.center.x - self.volume.size.x/2,self.volume.center.x + self.volume.size.x/2,Nx)
        self.rho_y = np.linspace(self.volume.center.y - self.volume.size.y/2,self.volume.center.y + self.volume.size.y/2,Ny)
    
    def __call__(self, p):
        weights, interp_idx = self.get_bilinear_row(p.x,p.y,self.rho_x,self.rho_y)
        return np.dot( self.rho_prime_vector[interp_idx], weights )                  

    def get_basis_jacobian(self,design_grid):
        # get array of grid points that correspond to epsilon vector
        #dj_deps = dj_deps.reshape(dj_deps.size,order='C')
        A = self.gen_interpolation_matrix(self.rho_x,self.rho_y,np.array(design_grid.x),np.array(design_grid.y))
        #return (eps.T * A).T
        return A
    
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