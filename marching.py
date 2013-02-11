import numpy as np
import pdb

def linear_interpolate(lo,hi,val=0):
  """
  Returns the fraction of the distance from the ``lo`` point to the ``hi``
  point at which the value ``val`` occurs.
  """
  # perturb the value if it intersects a vertex
  if lo==val: val = val + 1e-8*val + 1e-9
  if hi==val: val = val - 1e-8*val - 1e-9

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
  lows = np.where(ptvals<=val) #the interpolation will perturb equal values
  his  = np.where(ptvals>val)
  #for each vertex below the isovalue, compute the fraction of the distance
  # to vertices *above* the isovalue that the isovalue is at
  ivD = np.array([[linear_interpolate(l,h) for h in ptvals[his]] for l in ptvals[lows]])
  #return the low indices and relative distances from lows to isovalues
  return (ivD, lows,his)

def triangle_iso_area(pts,ivD,lows,his):
  """
  Given the vertices of a triangle, ``pts``, and the distances along the edges
  to the isovalue, ``ivDs``, compute the area of the *admissible* region
  inside the triangle.
  """
  #We want to find the area of the simplest region, either the admissible or
  # inadmissible side.  If the inadmissible side is easier, compute as if it was
  # the admissible side, but then later we need to flip the answer.
  #I.E., we invert if there are fewer HIGH vertices than LOW vertices, because
  # we want to work with as few "different" vertices as possible.
  #The shape for all HIGH vertices should be (0,3), but numpy makes this (0,).
  # Hence, we invert (i,j) if i<j or if j is not present.
  if len(ivD.shape)<2 or ivD.shape[0]>ivD.shape[1]:
    invert=True
    ivD = ivD.T
  else:
    invert=False
  
  #For a triangle, there are two cases for the intersection:
  # I  - all the vertices have the same sign
  #   Return the entire area of the triangle
  #
  # II - one vertex is different
  #   Return the area of the triangle bounded by the isosurface
  if 0 in ivD.shape:
    translated_verts = pts - pts[0]
    area = triangle_area(*translated_verts[1:])
  else:
    #Determine the different vertex.  If we haven't inverted, this will be the
    # sole LOW vertex.  If we DID invert, this will be the sole HIGH vertex.
    if invert:
      vtx0 = his[0]
    else:
      vtx0 = lows[0]
    #Now translate the triangle so the different vertex is at the origin
    pts0 = pts - pts[vtx0]
    #Remove the vertex at the origin
    pts1 = np.array([p for p in pts0 if p.any()])
    #Scale the lengths of the other edges to the isosurface distance
    pts2 = pts1*ivD
    #Compute the area of this triangle
    area = triangle_area(*pts2)

  #Uninvert if necessary, and return the area
  if invert:
    whole_tri = triangle_area(*(pts-pts[0])[1:])
    return whole_tri - area
  else:
    return area





