import numpy as np

def linear_interpolate(lo,hi,val=0):
  """
  Returns the fraction of the distance from the ``lo`` point to the ``hi``
  point at which the value ``val`` occurs.
  """
  return (val-lo)/(hi-lo)

def triangle_area(a,b):
  return 0.5*np.linalg.norm(a - (np.dot(a,b)/np.linalg.norm(b))*b)
