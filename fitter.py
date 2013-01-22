import numpy as np

def minor_determinants(mat):
  """
  Returns a list of principal minor determinants of the matrix
  ``mat``.
  """
  #TODO: Maybe this is premature optimization, but there's
  #      GOTTA be a way to use smaller minors to derive larger ones
  return [np.linalg.det(mat[:n,:n]) for n in range(1,mat.shape[0])]

def positive_definite_cost(mat):
  """
  Returns a measure of how far ``mat`` is from being positive definite.

  The measure is the magnitude of the sum of the negative minor determinants.

  If the matrix IS positive definite, it measures *how* positive definite
  it is by computing the sum of its *positive* minor determinants (all of
  them will be positive).  The answer in this case will be negative.
  """
  minors = minor_determinants(mat)
  indefinite = [-a for a in minors if a<0]
  definite   = [-a for a in minors if a>0]

  if len(indefinite)!=0: #the matrix is not positive definite
    return sum(indefinite)
  else: #the matrix *is* positive definite
    return sum(definite) 


