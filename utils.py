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

def test_matrix_params(*p):
  """
  Returns a symmetric matrix made from the given parameters.
  """
  (a,b,c) = p
  return np.array([[a,b],[b,c]])

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



