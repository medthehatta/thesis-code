"""
lintools.py

Some miscellaneous array / linear algebra tricks.
"""
import numpy as np
from itertools import count



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


def np_voigt_vec(A):
  """
  Same as np_voigt, but takes a list of tensors and does each.
  """
  rk = len(A.shape) - 1
  # this only works if each entry has positive, even rank
  if rk==0 or (rk%2)!=0: raise Exception("Argument's rank must be positive and even.")
  # get the dimension of the space
  dim = A.shape[-1]
  # this only works if it's square
  if not (np.array(A.shape)[1:]==dim).all(): raise Exception("Argument's indices must all have the same range.")
  # compute the "half-rank"; this will tell us how many indices we're combining
  rk2 = int(rk/2)
  # combine and return
  return np.reshape(A,(A.shape[0],dim**rk2,dim**rk2))


def symmetric(A):
  """
  Returns the symmetric part of A.
  """
  return (A + A.swapaxes(-1,-2))/2.

def antisymmetric(A):
  """
  Returns the antisymmetric part of A.
  """
  return (A - A.swapaxes(-1,-2))/2.


def direct_sum(A,B):
  """
  Returns the direct sum of two matrices.
  """
  return np.array(np.bmat([[A,np.zeros((A.shape[0],B.shape[1]))],
                          [np.zeros((B.shape[0],A.shape[1])),B]]))


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
  Return the tensor product of tensors A and B
  """
  return np.tensordot(A,B,0)


def matrix_norm(A):
  """
  Return the (squared) matrix norm: tr(AA^T)
  """
  return np.trace(np.dot(A,T.T))


def tensor_norm(A):
  """
  Return the (squared) tensor norm: A_ij...k A_ij...k
  """
  return np.tensordot(A,A,[range(len(A.shape))]*2)


def reorder_matrix(matrix,permutation):
  """
  Given a permutation of the basis vectors, reorder the matrix rows and
  columns to put the matrix in the reordered basis.
  """
  p = np.eye(matrix.shape[-1])[permutation]

  # Allow for either a list of matrices or just a matrix
  if len(matrix.shape)>2:
    P = np.tile(p,(matrix.shape[0],1,1))
  else:
    P = p

  return np.dot(P,np.dot(matrix,P.T))


def utri_flat(matrix):
  """
  Returns the upper triangle of a symmetric matrix in a flat list, ordered left
  to right.
  """
  nested = [matrix[i,i:].tolist() for i in range(len(matrix))]
  return np.array(sum(nested,[]))

def matrix_from_utri(flat,dim=3):
  """ 
  Returns a symmetric matrix made from the upper triangle ordered as in
  utri_flat
  """
  m = np.empty((dim,dim))
  for (k,(i,j)) in zip(count(),utri_indices(dim)):
    m[i,j]=flat[k]
    m[j,i]=m[i,j]
  return  m
  
def minor_dets(matrix):
  """
  Returns all the minor determinants.
  """
  return np.array([np.linalg.det(matrix[:i,:i]) for i in range(1,len(matrix)+1)])


def utri_indices(size):
  """
  Returns a flat list of the indices of an upper triangle from a matrix of
  shape ``size``x``size``.
  """
  return utri_flat(np.indices((size,size)).transpose(1,2,0))


def commutator(A,B,op=np.dot):
    return op(A,B) - op(B,A)

def anticommutator(A,B,op=np.dot):
    return op(A,B) + op(B,A)


def kronecker(A,B):
    return np.einsum('...ac,...bd->...abcd',A,B)

def cokronecker(A,B):
    return np.einsum('...ad,...bc->...abcd',A,B)

def symmetric_kronecker(A,B):
    return 0.5*(kronecker(A,B) + cokronecker(A,B))


    

