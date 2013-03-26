"""
stable_check.py

Find boundaries between the stable and unstable regions for a given model.
"""
import log_regression as reg
import numpy as np
import lintools as lin

def positive_definite_samples(model_D, deformation_map, lowcorner, hicorner, 
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
  unit_pts = np.random.random([num_pts]+list(lowcorner.shape))
                                     
  # Generate the points
  pts = lowcorner + (hicorner-lowcorner)*unit_pts
  deformations = deformation_map(pts)

  # Check positive-definiteness of tangent stiffnesses at each point
  tangent_stiffnesses = model_D(deformations, *params)
  # Convenience function for finding Voigt representation of a 4th rk tensor
  def voigt(d):
    return lin.reorder_matrix(lin.np_voigt(d),lin.VOIGT_ORDER)
  acceptable = np.array([lin.is_positive_definite(voigt(d)) for 
                         d in tangent_stiffnesses])

  # Return stable and unstable points in two separate arrays
  return (pts[acceptable==True,:],
          pts[acceptable==False,:])



