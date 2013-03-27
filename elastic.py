import numpy as np
import lintools as lin

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

  # Put in the triangle and then symmetrize
  N0 = np.array([[N11,2*N12,2*N13],[0,N22,2*N23],[0,0,N33]])
  N = (N0 + N0.T)/2.

  # The whole Voigt stiffness (but expanded to be fully 9x9) is given by
  shear_part = np.diagflat(shear_moduli)
  C = lin.direct_sum(N,lin.direct_sum(shear_part,shear_part))

  # Now we need to turn this into a bona-fide 4th rank tensor
  return lin.reorder_matrix(C,VOIGT_ORDER_INVERSE).reshape((3,3,3,3))


def isotropic_stiffness(E,n):
  """
  Given Young's modulus and Poisson's ratio, returns an isotropic stiffness
  tensor.
  """
  G = E/(2*(1+n)
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

VOIGT_ORDER = [0,4,8,5,6,1,7,2,3]
VOIGT_ORDER_INVERSE = [0,5,7,8,1,3,4,6,2]


