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

def monte_carlo_area_test(pts=None,lowcorner=np.zeros(2),hicorner=np.ones(2),numpts=None,ptsdensity=0.2,condition=field4):
  """
  Estimates the volume of the area in the rectangle given by ``lowcorner`` and
  ``hicorner`` which satisfies the vectorized condition ``condition``.
  Optionally, explicit points to test can be passed via ``pts``, or the number
  of samples to be taken in the rectangular region can be passed via
  ``numpts``.  If instead of ``numpts`` ``ptsdensity`` is specified, ``numpts``
  will be computed from the volume of the rectangular region.
  """
  if pts is None: 
    if numpts is None:
      numpts = ptsdensity*np.product(hicorner-lowcorner)
      if numpts<1: raise Exception("The domain is too small for a point density of {0}.  With this density, {1} points are being generated.  Increase the density or manually set numpts.".format(ptsdensity,numpts))
    pts = points_within_rectangle(lowcorner,hicorner,numpts)

  admissibles = condition(pts)
  return len(admissibles[admissibles==True])/len(pts)

