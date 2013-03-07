"""
strain.py

Strain modes.  These are parametrizations of typical useful deformation
gradients.
"""
import numpy as np

def simple_shear(gamma,e1=np.array([1,0,0]),e2=np.array([0,1,0])):
  """
  Returns simple shear of magnitude ``gamma`` in the plane spanned by ``e1``
  and ``e2``.
  """
  return np.eye(3) + gamma*np.outer(e1,e2)

def simple_extension(stretch,axis=np.array([1,0,0])):
  """
  Simple extension with stretch ratio ``stretch`` in direction ``axis``.
  """
  return np.eye(3) + (stretch-1)*np.outer(axis,axis)

def biaxial_extension(stretch1,stretch2,
                      e1=np.array([1,0,0]),e2=np.array([0,1,0])):
  """
  Biaxial extension with x-stretch ``stretch1`` and y-stretch ``stretch2``.
  """
  return np.eye(3) + (stretch1-1)*np.outer(e1,e1) + \
                     (stretch2-1)*np.outer(e2,e2)

def biaxial_extension_vec(stretches):
  """
  Biaxial extension function which supports our naiive calling from datafit.py
  """
  # We are going to add this deformation to the identity, so subtract off the
  # "undeformed" part of the stretches
  normalized = stretches - np.array([1.,1.])
  # Reshape the stretches so they can be multiplied by projection matrices
  normalized_r = normalized[:,:,None,None]

  # Compute the projection matrices along the fibers
  # The fibers are fixed for now as e1=i, e2=j
  axes = np.eye(3)[:2]
  # Projection operators are just the outer product of each axis with itself
  projections = np.einsum('...i,...j',axes,axes)

  # Compute the stretch matrices for each stretch
  stretch_operators = normalized_r*projections

  # Add these to the identity to get the total deformations
  return np.eye(3) + stretch_operators
  

  
