import numpy as np

def simple_shear(gamma,e1=np.array([1,0,0]),e2=np.array([0,1,0])):
  """
  Returns simple shear of magnitude ``gamma`` in the plane spanned by ``e1`` and
  ``e2``.
  """
  return np.eye(3) + gamma*np.outer(e1,e2)

def simple_extension(stretch,axis=np.array([1,0,0])):
  """
  Simple extension with stretch ratio ``stretch`` in direction ``axis``.
  """
  return np.eye(3) + (stretch-1)*np.outer(axis,axis)

