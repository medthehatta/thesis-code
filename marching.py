import numpy as np

def linear_interpolate(lo,hi,val=0):
  """
  Returns the fraction of the distance from the ``lo`` point to the ``hi``
  point at which the value ``val`` occurs.
  """
  # perturb the value if it intersects a vertex
  if lo==val: val = val + 1e-8*val
  if hi==val: val = val - 1e-8*val

  # make sure no divide by zero
  if hi!=lo:
    r = (val-lo)/(hi-lo)
    if r>0:
      return r

def triangle_area(a,b):
  """
  Returns the area of a triangle with one vertex at the origin and other two
  vertices at ``a`` and ``b``.
  """
  base = np.linalg.norm(b)
  perpendicular = a - (np.dot(a,b)/np.dot(b,b))*b
  height = np.linalg.norm(perpendicular)
  return 0.5*base*height

def iso_intersect_dists(ptvals,val=0):
  """
  Computes the positions at which the isosurface intersects the edges of a
  simplex whose vertices have values ``ptvals``.
  """
  #find vertices above and below isovalue
  lows = np.where(ptvals<val)
  his  = np.where(ptvals>val)
  #TODO (should check for isovalues at vertices)
  #for each vertex below the isovalue, compute the fraction of the distance
  # to vertices *above* the isovalue that the isovalue is at
  ivD = np.array([[linear_interpolate(l,h) for h in ptvals[his]] for l in ptvals[lows]])
  #return the low indices and relative distances from lows to isovalues
  return (lows,his, ivD)


