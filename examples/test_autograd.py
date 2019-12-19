from autograd import numpy as np
from autograd import grad
import meep_adjoint as mpa
import meep as mp
from matplotlib import pyplot as plt
np.random.seed(10)
import copy


#----------------------------------------------------------------------
#- autograd
#----------------------------------------------------------------------

'''def my_func(x):
    return np.sum(np.tanh(x))

dx = 1e-1
x0 = [1,2,3,4]
y0 = my_func
agrad = grad(my_func) 
avm = agrad(x0)
print(avm)

discrete_grad = [0,0,0,0]
for k in range(4):
    x1 = copy.deepcopy(x0)
    x1[k] -= dx
    x2 = copy.deepcopy(x0)
    x2[k] += dx

    f1 = my_func(x1)
    f2 = my_func(x2)

    discrete_grad[k] = (f2 - f1)/ (2*dx)

print(discrete_grad)

quit()'''

l_design = 1
waveguide_h = 0
fcen = 1
element_length = .1
element_type = 'CG 1'

#----------------------------------------------------------------------
#- design region and basis
#----------------------------------------------------------------------

design_size   = mp.Vector3(l_design, l_design, waveguide_h)
design_region = mpa.Subregion(fcen, name='design', center=mp.Vector3(), size=design_size)

basis = mpa.UniformBasis(region=design_region)

n = basis.dim
print(n)
beta_vector = 8
design_function = basis.parameterized_function(beta_vector)

N = 100
x = np.linspace(-l_design/2,l_design/2,N)
y = np.linspace(-l_design/2,l_design/2,N)
vecs = []
eps = np.zeros((N,N))
for nx,ix in enumerate(x):
    for ny,iy in enumerate(y):
        v = mp.Vector3(ix,iy)
        vecs.append(v)
        eps[nx,ny] = design_function(v)

grid = mpa.make_grid(size=[1,1],center=[0,0],dims=[N,N])
p_hat = basis.project(eps, grid=grid, differential=True)

print(beta_vector)
print(p_hat)

err = abs(beta_vector-p_hat)
print(err)
