import numpy as np
import lintools as lin

def manual_stiffness(normals,shears):
  """
  Returns a 4th rank stiffness tensor given the Voigt matrix for the normal
  stresses (the upper left block) and the Voigt matrix for the shears (the
  lower right block).
  """
  # The shears appear twice, because we want a 9x9, not 6x6 stiffness matrix
  shear_part = lin.direct_sum(shears,shears)
  voigt_mat = lin.direct_sum(normals,shear_part)
  return unvoigt(voigt_mat)


def orthotropic_stiffness(Ex,Ey,Ez,nyz,nzx,nxy,Gyz,Gzx,Gxy):
  """
  Returns an orthotropic stiffness tensor with the given parameters.
  """
  shear_moduli = 2*np.array([Gyz,Gzx,Gxy])
  
  # The formula for the stiffness tensor that I have uses all kinds of weird
  # versions of the poisson ratios, so we do the conversion first in order to
  # use the formula verbatim
  nzy = Ez/Ey*nyz
  nxz = Ex/Ez*nzx
  nyx = Ey/Ex*nxy

  # Another intermediate variable
  D = (1 - nxy*nyx - nyz*nzy - nzx*nxz - 2*nxy*nyz*nzx)/(Ex*Ey*Ez)

  # Construct the upper triangle of the "normal" stress block in Voigt
  # notation
  N11 = (1 - nzy*nzy)/(Ey*Ez*D)
  N12 = (nyx + nzx*nyz)/(Ey*Ez*D)
  N13 = (nzx + nyz*nzy)/(Ey*Ez*D)
  N22 = (1 - nzx*nxz)/(Ez*Ez*D)
  N23 = (nzy + nzx*nxy)/(Ez*Ex*D)
  N33 = (1 - nxy*nyz)/(Ex*Ey*D)

  # Put the triangle into a a real matrix
  N = np.array([[N11,N12,N13],[N12,N22,N23],[N13,N23,N33]])

  # The whole Voigt stiffness (but expanded to be fully 9x9) is given by
  shear_part = np.diagflat(shear_moduli)
  return manual_stiffness(N,shear_part)


def isotropic_stiffness(E,n):
  """
  Given Young's modulus and Poisson's ratio, returns an isotropic stiffness
  tensor.
  """
  G = E/(2*(1+n))
  return orthotropic_stiffness(E,E,E,n,n,n,G,G,G)


def voigt(A):
  """
  Returns the voigt-ified version of a 4th or 2nd rank tensor.
  (Uses a 9-component, not 6-component form for the 2nd rank tensors.)
  """
  return lin.reorder_matrix(lin.np_voigt(A),VOIGT_ORDER)

def voigt_vec(A):
  """
  Returns the voigt-ified version of a *list* of 4th or 2nd rank tensors.
  """
  return lin.reorder_matrix(lin.np_voigt_vec(A),VOIGT_ORDER)

def unvoigt(CC):
  """
  Double the rank of a voigt matrix to a 4th rank tensor, or vector -> matrix.
  """
  inrank = len(CC.shape)
  # Better be an even rank
  if (inrank%2) == 0:
    dim = np.sqrt(CC.shape[-1])
    # Better be a square number of rows/columns
    if int(dim)==float(dim):
      newshape = [dim]*(2*inrank)
      return lin.reorder_matrix(CC,VOIGT_ORDER_INVERSE).reshape(newshape)

VOIGT_ORDER = [0,4,8,5,6,1,7,2,3]
VOIGT_ORDER_INVERSE = [0,5,7,8,1,3,4,6,2]

def standard_orthotropic_P(e1=[1,0,0],e2=[0,1,0]):
    """
    Returns the single irreducible structure tensor for orthotropy given the
    unit normals of two of the orthogonal reflection planes.
    """
    return np.outer(e1,e1) - np.outer(e2,e2)

def orthotropic_projectors(P):
    """
    Given the irreducible orthotropic structure tensor, returns projection
    operators along normals to each of the planes of orthotropy.
    """
    I = np.eye(3)
    PP = np.dot(P,P)

    A1 = 0.5*(P + PP)
    A2 = 0.5*(P - PP)
    A3 = I - A1 - A2

    return np.array([A1,A2,A3])

