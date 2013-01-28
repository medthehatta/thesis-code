import numpy as np
import scipy.optimize as so
import random
from functools import reduce
import pdb

def normed(v):
  """
  Return the unit vector in the direction of ``v``
  """
  return v/np.linalg.norm(v)


def finite_gradient(f,x0=np.zeros((3,3)),dx=1e-5,basis=None):
  """
  Computes a finite-difference approximation to the gradient of a
  scalar-valued function ``f`` at the points in the list ``x0``.

  ``f`` should be vectorized.
  """ 
  # Default to the standard basis if none provided
  if basis is None:
    dimension = x0.shape[0]
    # the vectors of the standard basis are the rows of the identity matrix
    basis = np.eye(dimension)
  x0s       = np.tile(x0, (dimension,1))
  displaced = x0s + dx*basis
  # i hope dividing by a tiny number isn't going to give me issues
  return (f(displaced) - f(x0s))/dx


def finite_hessian(f,x0=np.zeros(3),dx=1e-5,basis=None):
 """
 Computes the hessian of a scalar-valued function ``f``.

 This might get really cumbersome really fast.
 """
 gradient = lambda x: finite_gradient(f,x,dx,basis)
 return finite_gradient(gradient,x0,dx,basis)


def gram_schmidt(basis):
  """
  Orthonormalizes the given ``basis`` of vectors.
  """
  def reduction_step(v,n): 
    return normed(v - np.dot(v,n)*n)

  def process_next_vector(basis_so_far,current_vector): 
    next_vector_result = reduce(reduction_step, basis_so_far, current_vector)
    if len(basis_so_far)!=0:
      return np.vstack([basis_so_far, next_vector_result])
    else:
      return np.array([next_vector_result])
  
  return reduce(process_next_vector, basis, [])
  

def hypersurface_basis(normal):
  """
  Return an orthonormal basis of the tangent space to the hypersurface
  orthogonal to ``normal``.
  """
  #TODO: if the normal is parallel to any of the standard basis elements,
  #      just return the current basis.

  # normalize the normal vector, if it isn't already
  lensqr = np.dot(normal,normal)
  if lensqr != 1:
    n = normal/np.sqrt(lensqr)
  else:
    n = normal

  dimension = len(normal)
  # The vectors of the standard basis are the rows of the identity matrix.
  # We want an orthonormal basis with one of the vectors in the direction
  #   of the normal so all the *other* basis elements are parallel to the
  #   hypersurface
  # Start by replacing the first basis vector in the set with the normal, and
  #   then orthonormalize
  skewed_basis = np.vstack([n,
                            np.eye(dimension)[1:]])
  return gram_schmidt(skewed_basis)


def naive_seed_isosurface(func,dim,maxiters=1e4,spread=10):
  """
  Given a continuous scalar function ``func`` on a vector space and ``dim``,
  the space's dimension, finds a seed point for one connected component of the
  zero isosurface.

  We do this by randomly sampling points in the domain until two have different
  signs.  Then we root-find along the line between the points until we find the
  zero.  This will be a seed point of the isosurface.

  Of course this will only find one connected component of the isosurface.  To
  find all the components, we need a more sophisticated algorithm.
  """
  def random_sign(): return np.array([random.choice([-1,1]) for x in range(dim)])

  # Select the first point and check the sign of the function
  pt1  = random_sign()*spread*np.random.random(dim)
  sgn1 = np.sign(func(pt1))

  # Rejection sample for the second point until we have the other sign
  pt2 = random_sign()*spread*np.random.random(dim)
  while np.sign(func(pt2))==sgn1 and maxiters:
    pt2 = random_sign()*spread*np.random.random(dim)
    maxiters-=1

  # Failed to converge quickly enough
  if maxiters<1: return None

  # Once we have the second point, parametrize the segment between the two
  if sgn1<0:
    chord = lambda s: pt1 + s*(pt2-pt1)
  else:
    chord = lambda s: pt2 + s*(pt1-pt2)

  # Now root find along the segment between them
  t0 = so.zeros.brenth(lambda t: func(chord(t)),0,1)

  # Return the seed point
  return chord(t0)

