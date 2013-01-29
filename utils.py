import numpy as np
import fitter as fit
import propagate as prop
import scipy.optimize as so
import operator
from imp import reload
from functools import reduce
from itertools import islice, count, takewhile
import pdb

def random_symmetric_matrix(shape):
  """
  Returns a random symmetric matrix with the given shape
  """
  M = np.random.random(shape)
  return (M.T + M)/2

def ireversed(iterator):
  """
  Reverses an iterator (which usually doesn't work) and makes the result into a
  list.
  """
  return list(reversed(list(iterator)))

def direct_sum(A,B):
  """
  Returns the direct sum of two matrices.
  """
  return np.bmat([[A,np.zeros((A.shape[0],B.shape[1]))],
                  [np.zeros((B.shape[0],A.shape[1])),B]])

def make_symmetric_matrix(*p):
  """
  Returns a symmetric matrix from a vector of components.

  If the vector has a nontriangular number of components in it, put ones that
  don't fit on the diagonal.
  """
  pr = reversed(p)
  all_diagonals = (ireversed(islice(pr,i)) for i in count(1))
  diagonals = ireversed(takewhile(operator.truth, all_diagonals))

  diag_mats = [np.diagflat(d,i) for (d,i) in zip(diagonals,count())]
  # Check to make sure all the diagonals fit properly into the matrix.  If they
  #   didn't, it just means that the list we were given doesn't have a triangular
  #   number of elements.  In that case, put the extra elements (from the front)
  #   on the diagonal.
  if diag_mats[0].shape == diag_mats[1].shape:
    utri = sum([np.diagflat(d,i) for (d,i) in zip(diagonals,count())])
  else:
    utri0 = sum([np.diagflat(d,i) for (d,i) in zip(diagonals[1:],count())])
    utri  = direct_sum(np.diagflat(diagonals[0]), utri0)

  return (utri + utri.T)/2

def perturb_array(A,scale=1):
  """
  Slightly perturb the values of an array with displacements of order
  ``scale``.
  """
  sign = np.random.choice([-1,1],size=A.shape)
  magnitude = (scale*10) #default scale is ~10^(-1)
  return A + magnitude*sign*np.random.random(A.shape)

def field1(x):
  """
  The positive-definite cost of a symmetric matrix built from a vector of
  components
  """
  return fit.positive_definite_cost(make_symmetric_matrix(*x))

def field2(XX):
  """
  A fake vectorized version of field1.
  Eventually I should *actually* vectorize this, but I have no idea how.
  """
  return np.array([[field1(x) for x in X] for X in XX])


