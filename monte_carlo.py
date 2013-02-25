"""
monte_carlo.py

Monte carlo volume estimation: 
 - This is really just Monte Carlo integration of the density f=1
"""
import numpy as np

def points_within_rectangle(lowcorner=np.zeros(2), hicorner=np.ones(2), numpts=50):
  """
  Uniformly generate points in a rectangular box (product of intervals).
  """
  return lowcorner + np.random.random([numpts]+list(lowcorner.shape))*(hicorner-lowcorner)

def area_test(pts=None,lowcorner=np.zeros(2),hicorner=10*np.ones(2),numpts=None,ptsdensity=0.2,condition=None):
  """
  Estimates the volume fraction of the area in the rectangle given by
  ``lowcorner`` and ``hicorner`` which satisfies the vectorized condition
  ``condition``.  
  Optionally, explicit points to test can be passed via ``pts``, or the number
  of samples to be taken in the rectangular region can be passed via
  ``numpts``.  
  If instead of ``numpts`` ``ptsdensity`` is specified, ``numpts`` will be
  computed from the volume of the rectangular region.
  """
  # if the user didn't pass a condition, just make it true
  if condition is None:
    # every point is true
    condition = lambda x: np.array([True]*len(x))
  # if we don't have specified sample points, generate them randomly
  if pts is None: 
    # if we don't have a number of points, use the point density
    if numpts is None:
      numpts = ptsdensity*np.product(hicorner-lowcorner)
    # we need at least one sample point
    if numpts<1: 
      exception_text = "The domain is too small for a point density of {0}.  With this density, {1} points are being generated.  Increase the density or manually set numpts."
      raise Exception(exception_text.format(ptsdensity,numpts))
    pts = points_within_rectangle(lowcorner,hicorner,numpts)
  # get true or false for each point in pts
  admissibles = condition(pts)
  # return the fraction which are true
  return len(admissibles[admissibles])/len(pts)

