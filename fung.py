"""
fung.py

Strain energy density and tangent stiffness for the Fung tissue model.
"""
import lintools as lin
import numpy as np
import pdb

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
  return lin.reorder_matrix(C,lin.VOIGT_ORDER).reshape((3,3,3,3))


def model(F,*params):
  """
  Exposed fung_P with fully orthotropic stiffness.
  """
  return fung_P(F,params[0],orthotropic_stiffness(*params[1:]))


def model_isotropic(F,*params):
  """
  Exposed fung_P with isotropic stiffness.
  Params are Young's modulus and Poisson's ratio.
  """
  (E,n) = params
  G = E/(2*(1+n))
  return fung_P(F,params[0],orthotropic_stiffness(E,E,E,n,n,n,G,G,G))


def fung(F,c,CC):
  """
  Returns the strain energy given the deformation, ``F``, and the model
  parameters.
  W = c/2 (exp(Q) - 1)
  Q = E:CC:E
  """
  dim = F.shape[-1]
  E = 0.5*np.dot(F.T,F) - 0.5*np.eye(dim)
  Q = np.einsum('abcd,ab,cd',CC,E,E)
  return 0.5*c*(np.exp(Q) - 1)

def fung_P(F,c,CC):
  """
  Returns the PK1 stress for the given deformation ``F`` and the model
  parameters.
  """
  dimension = F.shape[-1]
  if len(F.shape)>2:
    I = np.tile(np.eye(dimension),(F.shape[0],1,1))
    CC = np.tile(CC,(F.shape[0],1,1,1,1))
  else:
    I = np.eye(dimension)
  E = 0.5*np.einsum('...ab,...ac',F,F) - 0.5*I
  Q = np.einsum('...abcd,...ab,...cd',CC,E,E)

  # Derivative of Q wrt E: (rank 2)
  dQdE = 2*np.einsum('...abcd,...cd',CC,E)
  # Derivative of E wrt F: (rank 4)
  dEdF = 0.5*np.einsum('...ab,...cd->...adcb',I,F) + 0.5*np.einsum('...ab,...cd->...bdca',I,F)
  # Derivative of Q wrt F: (dQdE)ij (dEdF)ijkl (rank 2)
  dQdF = np.einsum('...ij,...ijkl',dQdE,dEdF)
  # Return value: c/2 exp(Q) dQdF
  # (numpy needs dQdF.T to properly multiply with np.exp(Q))
  return 0.5*c*(np.exp(Q)*dQdF.T).T

def fung_D(F,c,CC):
  """
  Returns the tangent stiffness given the deformation, ``F``, and the model
  parameters.
  """
  dimension = F.shape[-1]
  I = np.eye(dimension)
  E = 0.5*np.dot(F.T,F) - 0.5*I
  Q = np.einsum('abcd,ab,cd',CC,E,E)

  # Derivative of Q wrt E: (rank 2)
  dQdE = 2*np.einsum('abcd,cd',CC,E)
  # Derivative of E wrt F: (rank 4)
  dEdF = 0.5*(np.einsum('ab,cd->adcb',I,F) + np.einsum('ab,cd->bdca',I,F))
  # Derivative of Q wrt F: (dQdE)ij (dEdF)ijkl (rank 2)
  dQdF = np.einsum('ij,ijkl',dQdE,dEdF)

  # Second derivative of Q wrt E: (rank 4)
  ddQdEdE = 2*CC
  # Second derivative of E wrt F: (rank 6)
  ddEdFdF = 0.5*(np.einsum('ij,kl,mn->kmilnj',I,I,I) + 
                 np.einsum('ij,kl,mn->kmjlni',I,I,I))

  # Each term of 3-part product rule
  ddW1 = lin.tensor(dQdF,dQdF)
  ddW2 = np.einsum('ijkl,ijmn,klpq',ddQdEdE,dEdF,dE,dF)
  ddW3 = np.einsum('ij,ijklmn',dQdE,ddEdFdF)
  # Return the sum of these, all times c/2 exp(Q)
  return 0.5*c*np.exp(Q)*(ddW1 + ddW2 + ddW3)

