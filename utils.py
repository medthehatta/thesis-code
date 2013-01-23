import numpy as np
import fitter as fit
import propagate as prop
import scipy.optimize as so
from imp import reload
from functools import reduce
import pdb

def random_symmetric_matrix(shape):
  """
  Returns a random symmetric matrix with the given shape
  """
  M = np.random.random(shape)
  return (M.T + M)/2


def test_matrix_params(*p):
  """
  Returns a symmetric 2x2 matrix from the specified parameters.
  """
  (a,b,c) = p
  return np.array([[a,b],[b,c]])


