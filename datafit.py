"""
datafit.py

Routines for performing the constrained fit on stress/strain data.
"""
import numpy as np
import lintools as lin

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

def data_leastsqr_fn(Fs,Ps,model):
  """
  Returns ``data_leastsqr`` as a function of parameters.
  Simple wrapper for use with minimization algorithms.
  """
  return lambda p: data_leastsqr(Fs,Ps,model,*p)

