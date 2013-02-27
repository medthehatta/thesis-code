"""
datafit.py

Routines for performing the constrained fit on stress/strain data.
"""
import numpy as np
import lintools as lin
import monte_carlo

def data_leastsqr(Fs,Ps,model,*params):
  """
  Returns the least square cost of a parameter choice for a given model.

  The model passed to this should be vectorized to take a list of deformation
  gradients and provide a list of PK1 stresses.
  """
  # compute all the deviations
  deviations = Ps - model(Fs,*params)

  # this might be a squared sum of squared deviations, but it shouldn't matter
  # as long as it's monotone and not too steep
  return lin.tensor_norm(deviations)

def penalty_admissible_region(lowcorner,hicorner,condition,*params,ptsdensity=0.2):
  """
  Returns an estimate of the inadmissible volume in the box whose extreme
  corners are at ``lowcorner`` and ``hicorner``.
  ``condition`` takes an array of primary arguments and a fixed parameter
  vector and returns an array of booleans saying whether the primary arguments
  satisfied the constraint.
  """
  numpts = ptsdensity*np.product(hicorner-lowcorner) 
  if numpts<1: raise Exception("Domain too small for point density {0}".format(ptsdensity))
  unit_pts = np.random.random((numpts,len(lowcorner)))
  pts = lowcorner + (hicorner-lowcorner)*unit_pts
  admissibles = condition(pts,*params)
  return len(admissibles[admissibles])/len(pts)

def admissible_penalty_cost(Fs,Ps,model,lowcorner,hicorner,condition,ptsdensity=0.2):
  """
  Returns ``data_leastsqr`` as a function of parameters.
  Simple wrapper for use with minimization algorithms.
  """
  return lambda p: data_leastsqr(Fs,Ps,model,*p) + penalty_admissible_region(lowcorner,hicorner,condition,*p,ptsdensity=ptsdensity)
