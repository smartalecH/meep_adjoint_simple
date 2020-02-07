"""Definition of the Basis abstract base class.

    A Basis is a finite-dimensional space of scalar functions defined on
    a finite spatial region (the *domain*).

    **Instance Data**

    An instance of Basis is defined by the following data:
        (1) a domain V,
        (2) a set of D scalar basis functions {b_n(x)}, n=0,1,...,D-1, defined in V.

    Once item (2) is specified, individual elements f(x) in the space f(x) are
    identified by a D-dimensional vector of expansion coefficients
    {\beta_n} according to f(x) = \sum \beta_n b_n(x).

    **Exported Methods**

       An instance of Basis exports methods implementing the following two operations:

          (1) *Gradient*: Given a vector representing the output of the basis, multiply it
              by the gradient of the basis.

          (2) *Function instance*: Given a set of expansion coefficients {beta_n}
              return a callable func that inputs a spatial variable x and
              outputs func(x) = \sum beta_n b_n(x).
"""

from numbers import Number
import sys
from sympy import lambdify, Symbol
import re
import numpy as np
import meep as mp

from . import Subregion

class GridFunc(object):
    """Given a grid of spatial points {x_n} and a scalar function of a
       single spatial variable f(x) (whose specification may take any
       of several possible forms), return a scalar function of a
       single integer GridFunc(n) defined by GridFunc(n) == f(x_n).

    Arguments
    ---------
        f: function-like
           specification of function f(x)

        grid: array-like
           grid of points {x_n} for integers n=0,1,...,

    Returns
    -------
        GridFunc (callable) satisfying GridFunc(n)==f(x_n).
    """

    def __init__(self,f,grid):
        self.p=grid.points
        self.fm=self.fv=self.ff=None
        if isinstance(f,np.ndarray) and f.shape==grid.shape:
            self.fm = f.flatten()
        elif isinstance(f,Number):
            self.fv = f
        elif callable(f):
            self.ff = lambda n: f(self.p[n])
        elif isinstance(f,str):
            ffunc=lambdify( [Symbol(v) for v in 'xyz'],f)
            self.ff = lambda n:ffunc(self.p[n][0],self.p[n][1],self.p[n][2])
        else:
            raise ValueError("GridFunc: failed to construct function")

    def __call__(self, n):
        return self.fm[n] if self.fm is not None else self.fv if self.fv is not None else self.ff(n)


######################################################################
#invoke python's 'abstract base class' formalism in a version-agnostic way
######################################################################
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {'__slots__': ()}) # compatible with Python 2 and 3


#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Basis is the abstract base class from which classes describing specific
# basis sets should inherit.
#----------------------------------------------------------------------
#----------------------------------------------------------------------
class Basis(ABC):
    """
    """

    def __init__(self, dim, region=None, size=None, center=mp.Vector3(), offset=0.0):
        self.dim, self.offset = dim, offset
        self.region = region if region else Subregion(center=center,size=size)

    @property
    def dimension(self):
        return self.dim

    @property
    def domain(self):
        return self.region

    @property
    def names(self):
        return [ 'b{}'.format(n) for n in range(self.dim) ]

    ######################################################################
    # get full vector of basis-function values at a single evaluation point
    #  (pure virtual method, must be overriden by subclasses)
    ######################################################################
    @abstractmethod
    def get_bvector(self, p):
        raise NotImplementedError("derived class must implement get_bvector()")

