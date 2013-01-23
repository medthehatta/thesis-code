import numpy as np
from functools import reduce

def normed(v):
  """
  Return the unit vector in the direction of ``v``
  """
  return v/np.linalg.norm(v)


def finite_gradient(f,x0=np.zeros(3),dx=1e-5,basis=None):
  """
  Computes a finite-difference approximation to the gradient of a
  scalar-valued function ``f`` at the point ``x0``.
  """ 
  # Default to the standard basis if none provided
  if not basis:
    dimension = len(x0)
    # the vectors of the standard basis are the rows of the identity matrix
    basis = np.eye(dimension)
  # i hope dividing by a tiny number isn't going to give me issues
  return np.array([f(x0 + dx*ei)/dx for ei in basis])


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



