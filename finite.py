"""
finite.py

Some routines for finite-difference numerics.

This probably duplicates some functionality from scipy or numpy, but I don't
want to spend too much time digging through docs.
"""
import numpy as np


def finite_gradient(f,x0=np.zeros(3),dx=1e-5):
  """
  Computes a finite-difference approximation to the gradient of a
  vector-valued function ``f`` at the points in the list ``x0``.

  ``f`` should be vectorized.
  """ 
  # Allow passing of single vectors for x0, but then they need to be converted
  #  to a singleton list
  if len(x0.shape)==1: x0 = np.array([x0])

  # the vectors of the standard basis are the rows of the identity matrix
  dimension = x0.shape[-1]
  basis     = np.tile(np.eye(dimension), (x0.shape[0],1,1))
  x0s       = np.tile(x0, (dimension,1,1)).transpose(1,0,2)
  displaced = x0s + dx*basis
  # i hope dividing by a tiny number isn't going to give me issues
  return (f(displaced) - f(x0s))/dx


def finite_hessian(f,x0=np.zeros(3),dx=1e-5):
 """
 Computes the hessian of a scalar-valued function ``f``.

 This might get really cumbersome really fast.
 """
 gradient = lambda x: finite_gradient(f,x,dx)
 return finite_gradient(gradient,x0,dx)

