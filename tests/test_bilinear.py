import meep as mp
import meep_adjoint as mpa
import numpy as np
import unittest

class TestBilinearInterpolationMatrix(unittest.TestCase):
    def run_test(self,resolution,Nx,Ny):
        Sx = 1
        Sy = 1

        z = lambda x,y: 5*np.abs((x**3 + y**2 + x**2 * y)) + 1

        cell_size = mp.Vector3(Sx,Sy)

        design_size   = mp.Vector3(Sx, Sy, 0)
        design_region = mpa.Subregion(name='design', center=mp.Vector3(), size=design_size)
        basis = mpa.BilinearInterpolationBasis(region=design_region,Nx=Nx,Ny=Ny)

        rho_x = np.linspace(-Sx/2,Sx/2,Nx)
        rho_y = np.linspace(-Sy/2,Sy/2,Ny)
        rho_X, rho_Y = np.meshgrid(rho_x,rho_y)
        rho = z(rho_X,rho_Y)
        beta_vector = rho.reshape(rho.size,order='C')

        design_function = basis.parameterized_function(beta_vector)
        design_object = [mp.Block(center=design_region.center, size=design_region.size, epsilon_func = design_function.func())]

        geometry = design_object

        sim = mp.Simulation(
                        cell_size=cell_size,
                        geometry=geometry,
                        resolution=resolution)
        sim.init_sim()
        (x_interp,y_interp,z_interp,w) = sim.get_array_metadata(center=mp.Vector3(), size=cell_size)
        eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
        
        rho_X_interp, rho_Y_interp = np.meshgrid(x_interp,y_interp)
        rho_interp_true = z(rho_X_interp,rho_Y_interp)

        A = mpa.gen_interpolation_matrix(rho_x,rho_y,np.array(x_interp),np.array(y_interp))
        rho_interp_hat = A * rho.reshape(rho.size,order='C')

        # Calculate errors
        sim_err = np.mean(np.abs(eps_data.reshape(eps_data.size,order='C') - rho_interp_true.reshape(rho_interp_true.size,order='C'))**2)
        A_err = np.mean(np.abs(rho_interp_hat - rho_interp_true.reshape(rho_interp_true.size,order='C'))**2)
        meep_err = np.mean(np.abs(rho_interp_hat - eps_data.reshape(eps_data.size,order='C'))**2)
        
        print("Simulation Mean Squared Error:                             ",sim_err)
        print("Interpolation Matrix Mean Squared Error:                   ",A_err)
        print("MSE between interp. matrix and meep's dielectric function: ",meep_err)

        '''from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(eps_data)
        plt.title('Bilinear Interpolation')

        plt.figure()
        plt.imshow(rho_interp_hat.reshape(rho_interp_true.shape,order='C'))
        plt.title('Sparse Matrix Solution')

        plt.figure()
        plt.imshow(rho_interp_true)
        plt.title('Analytic Solution')

        plt.show()'''

        return sim_err, A_err, meep_err
    
    def test_matched_interpolations(self):
        sim_err, A_err, meep_err = self.run_test(resolution=50,Nx=50,Ny=50)
        self.assertLess(sim_err, 1e-5)
        self.assertLess(A_err, 1e-5)
        self.assertLess(meep_err, 1e-5)

    def test_unmatched_interpolation(self):
        sim_err, A_err, meep_err = self.run_test(resolution=50,Nx=5,Ny=5)
        self.assertLess(sim_err, 1e-2)
        self.assertLess(A_err, 1e-2)
        self.assertLess(meep_err, 1e-5)

if __name__ == '__main__':
    unittest.main()
    
    