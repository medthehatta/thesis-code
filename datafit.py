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

def positive_definite_penalty(model_D, deformation_map, lowcorner, hicorner, 
                              *params, ptsdensity=0.2):
  """
  The penalty for a model having a non-positive-definite tangent stiffness over
  some portion of the deformation domain which we require to have
  positive-definite tangent stiffnesses.
  
  Arguments
  -----------
  - ``model_D`` - Function taking deformation and model parameters to tangent
    stiffness
  - ``deformation_map`` - Function taking deformation parameters to
    deformations
  - ``lowcorner`` - The low corner of the rectangular desired deformation
    parameter region 
  - ``hicorner`` - The high corner of the rectangular desired deformation
    parameter region
  - ``params`` - The model parameters being evaluated
  - ``ptsdensity`` - Density of points to sample with
  """
  # Generate sample points with given density
  num_pts = ptsdensity*np.product((hicorner-lowcorner).ravel())
  if num_pts < 5:
    raise Exception("Point density not high enough, or admissible domain too",
                    "small.")
  # This generates ``numpts`` random numbers between 0 and 1 and reshapes the
  # array so it can be multiplied by the difference between ``lowcorner`` and
  # ``hicorner``.
  unit_pts = np.random.random(num_pts).reshape([num_pts] + \
                                               [1]*len(lowcorner.shape))
  # Generate the points
  pts = lowcorner + (hicorner-lowcorner)*unit_pts
  deformations = deformation_map(pts)

  # Check positive-definiteness of tangent stiffnesses at each point
  tangent_stiffnesses = model_D(deformations, *params)
  acceptable = np.array([lin.is_positive_definite(lin.np_voigt(d)) for 
                         d in tangent_stiffnesses])

  # Return lambda * the fraction of sampled points which were positive-definite
  num_acceptable = acceptable[acceptable].shape[0]
  return num_acceptable/num_pts

