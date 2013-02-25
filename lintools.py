"""
lintools.py

Some miscellaneous array / linear algebra tricks.
"""
import numpy as np


def np_voigt(A):
  """
  NumPy Voigt
  Compute the Voigt version of a higher-rank tensor, but with sequential
  ordering of the sets of indices.
  """
  rk = len(A.shape)
  # this only works if it has positive, even rank
  if rk==0 or (rk%2)!=0: raise Exception("Argument's rank must be positive and even.")
  # get the dimension of the space
  dim = A.shape[0]
  # this only works if it's square
  if not (np.array(A.shape)==dim).all(): raise Exception("Argument's indices must all have the same range.")
  # compute the "half-rank"; this will tell us how many indices we're combining
  rk2 = int(rk/2)
  # combine and return
  return np.reshape(A,(dim**rk2,dim**rk2))


def antisymmetric(A):
  """
  Returns the antisymmetric part of A.
  """
  return (A - A.T)/2.


def direct_sum(A,B):
  """
  Returns the direct sum of two matrices.
  """
  return np.bmat([[A,np.zeros((A.shape[0],B.shape[1]))],
                  [np.zeros((B.shape[0],A.shape[1])),B]])


def perturb_array(A,scale=1):
  """
  Slightly perturb the values of an array with displacements of order
  ``scale``.
  """
  sign = np.random.choice([-1,1],size=A.shape)
  magnitude = (scale*10) #default scale is ~10^(-1)
  return A + magnitude*sign*np.random.random(A.shape)


def is_positive_definite(mat):
  """
  Test a matrix for positive-definiteness.
  """
  try:
    np.linalg.cholesky(mat)
    return True
  except np.linalg.LinAlgError:
    return False


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
  

def normed(v):
  """
  Return the unit vector in the direction of ``v``
  """
  return v/np.linalg.norm(v)


def tensor(A,B):
  """
  Return the tensor product of matrices A and B
  """
  return np.einsum('ij,kl->ijkl',A,B)


def matrix_norm(A):
  """
  Return the matrix norm: tr(AA^T)
  """
  return np.trace(np.dot(A,T.T))
