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
  return 0.5*np.linalg.norm(a - (np.dot(a,b)/np.linalg.norm(b))*b)

def tri1(A,B,C,val=0,a=np.zeros(2),b=np.array([1,0]),c=np.array([0.5,np.sqrt(0.75)])):
  """
  Takes vertex values (with vertices on an equilateral triangle with one edge
  parallel to the x-axis) and returns the vertices of the triangular region in
  the partition by the isovalue.
  """
  vert = np.array([a,b,c])
  disp = np.roll(vert,-1,axis=0) - vert
  terp = [linear_interpolate(v,w,val) for (v,w) in [(A,B),(B,C),(C,A)]]
  terp0 = [(x or 0) for x in terp]
  terpl = np.tile(terp0,(vert.shape[-1],1)).T
  return vert+terpl*disp

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
    

    
