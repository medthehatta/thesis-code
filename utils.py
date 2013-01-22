import numpy as np
import fitter as fit
import scipy.optimize as so

def random_symmetric_matrix(shape):
  """
  Returns a random symmetric matrix with the given shape
  """
  M = np.random.random(shape)
  return (M.T + M)/2




