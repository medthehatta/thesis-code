import numpy as np
import fitter as fit
import propagate as prop
import scipy.optimize as so
from scipy.spatial import Delaunay
import operator
import marching
import matplotlib.pyplot as plt
from imp import reload
from functools import reduce
from itertools import islice, count, takewhile
import pdb

def random_symmetric_matrix(shape):
  """
  Returns a random symmetric matrix with the given shape
  """
  M = np.random.random(shape)
  return (M.T + M)/2

def np_voigt(A):
  """
  NumPy Voigt
  Compute the Voigt version of a higher-rank tensor, but with sequential
  ordering of the sets of indices.
  """
  rk = len(A.shape)
  # this only works if it has positive, even rank
  if rk==0 or (rk%2)!=0: raise Exception("Argument's rank must be positive and even.")
  # get the dimension of the space
  dim = A.shape[0]
  # this only works if it's square
  if not (np.array(A.shape)==dim).all(): raise Exception("Argument's indices must all have the same range.")
  # compute the "half-rank"; this will tell us how many indices we're combining
  rk2 = int(rk/2)
  # combine and return
  return np.reshape(A,(dim**rk2,dim**rk2))

def is_symmetric(A):
  """
  Checks if the even-rank tensor A has major symmetry.
  """
  Anv = np_voigt(A)
  return Anv == Anv.T

def antisymmetric(A):
  """
  Returns (twice) the antisymmetric part of A.
  """
  return A - A.T

def ireversed(iterator):
  """
  Reverses an iterator (which usually doesn't work) and makes the result into a
  list.
  """
  return list(reversed(list(iterator)))

def direct_sum(A,B):
  """
  Returns the direct sum of two matrices.
  """
  return np.bmat([[A,np.zeros((A.shape[0],B.shape[1]))],
                  [np.zeros((B.shape[0],A.shape[1])),B]])

def make_symmetric_matrix(*p):
  """
  Returns a symmetric matrix from a vector of components.

  If the vector has a nontriangular number of components in it, put ones that
  don't fit on the diagonal.
  """
  pr = reversed(p)
  all_diagonals = (ireversed(islice(pr,i)) for i in count(1))
  diagonals = ireversed(takewhile(operator.truth, all_diagonals))

  diag_mats = [np.diagflat(d,i) for (d,i) in zip(diagonals,count())]
  # Check to make sure all the diagonals fit properly into the matrix.  If they
  #   didn't, it just means that the list we were given doesn't have a triangular
  #   number of elements.  In that case, put the extra elements (from the front)
  #   on the diagonal.
  if diag_mats[0].shape == diag_mats[1].shape:
    utri = sum([np.diagflat(d,i) for (d,i) in zip(diagonals,count())])
  else:
    utri0 = sum([np.diagflat(d,i) for (d,i) in zip(diagonals[1:],count())])
    utri  = direct_sum(np.diagflat(diagonals[0]), utri0)

  return (utri + utri.T)/2

def perturb_array(A,scale=1):
  """
  Slightly perturb the values of an array with displacements of order
  ``scale``.
  """
  sign = np.random.choice([-1,1],size=A.shape)
  magnitude = (scale*10) #default scale is ~10^(-1)
  return A + magnitude*sign*np.random.random(A.shape)

def field1(x):
  """
  The positive-definite cost of a symmetric matrix built from a vector of
  components
  """
  return fit.positive_definite_cost(make_symmetric_matrix(*x))

def field2(XX):
  """
  A fake vectorized version of field1.
  Eventually I should *actually* vectorize this, but I have no idea how.
  """
  return np.array([[field1(x) for x in X] for X in XX])

def field3(XX):
  return np.array([field1(x) for x in XX])

def field4(XX):
  return np.array([fit.is_positive_definite(make_symmetric_matrix(*x)) for x in XX])

def draw_triangle(pts,vals=[0,0,0],labels=count()):
  """
  Plots a triangle with colored vertices
  """
  plt.scatter(pts.T[0],pts.T[1],c=plt.cm.bwr_r(10*vals),vmin=-10,vmax=10,s=120)
  plt.plot(pts.T[0].tolist()+[pts.T[0][0]], pts.T[1].tolist()+[pts.T[1][0]],'k-')

  for (i,p) in zip(labels,pts):
    plt.text(p[0]+0.05,p[1]+0.05,str(i),fontsize=14,weight='bold')

def points_within_rectangle(lowcorner=np.zeros(2), hicorner=np.ones(2), numpts=50):
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
    pts = points_within_rectangle(lowcorner,hicorner,numpts)

  admissibles = condition(pts)
  if len(admissibles)==0:
    return -1
  else:
    return len(admissibles[admissibles==True])/len(admissibles)

def area_monte_carlo_triangles_test(pts=None,hirect=np.ones(2),lorect=np.array([1e-5,1e-5]),numpts=0):
  """
  Finds the area of the admissible region in a given rectangle.
  Assumptions:
  - isovalue is 0
  - fitness field is field2
  """
  #clear the plot display
  plt.cla()

  #if points provided, just use those.  otherwise, compute
  if pts is None:
    #mesh the rectangle, adding ``numpts`` interior points if desired
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    samples = lorect + np.random.random((numpts,2))*(hirect-lorect)
    pts = np.vstack([lorect,
                     hirect,
                     e1*lorect+e2*hirect,
                     e2*lorect+e1*hirect,
                     samples])
  #triangulate the point cloud
  triangulation = Delaunay(pts)

  #compute the admissible function for the vertices
  vals = field2([pts])[0]

  #visit each triangle in the triangulation and compute the admissible area
  sum_area = 0
  sum_triarea = 0
  for T in triangulation.vertices:
    tpts           = np.array([pts[i] for i in T])
    tvals          = np.array([vals[i] for i in T])
    (ivD,lows,his) = marching.iso_intersect_dists(tvals)
    area           = marching.triangle_iso_area(tpts,ivD,lows,his)
    triarea        = marching.triangle_area(*(tpts-tpts[0])[1:])
    sum_area+=area
    sum_triarea+=triarea

    #print debug info
    global_lows = [T[i] for i in lows[0]]
    global_his  = [T[i] for i in his[0]]
    if len(global_lows)==0 or len(global_his)==0:
      printable_ivD = np.array([[]])
    else:
      printable_ivD  = np.vstack([[-111]+global_his, 
                                  np.hstack([np.transpose([global_lows]),ivD])])
    print("\nTriangle: {0}\nDistances: {1}\n{2}\nArea: {3}%".format(T,ivD.shape,printable_ivD,100*area/triarea))
    #add a triangle to the plot
    draw_triangle(tpts,tvals,labels=T)

  #display the triangulation
  plt.show()

  return {'points':pts, 'values':vals, 'admissible area':sum_area, 'total area':sum_triarea, 'area ratio':(sum_area/sum_triarea)}


  

