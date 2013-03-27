"""
fung.py

Strain energy density and tangent stiffness for the Fung tissue model.
"""
import lintools as lin
import elastic as el
import numpy as np
import pdb

def model(F,*params):
  """
  Exposed fung_P with fully orthotropic stiffness.
  """
  return fung_P(F,params[0],el.orthotropic_stiffness(*params[1:]))

def model_D(F,*params):
  """
  Exposed fung_D with fully orthotropic stiffness.
  """
  return fung_D(F,params[0],el.orthotropic_stiffness(*params[1:]))


def model_isotropic(F,*params):
  """
  Exposed fung_P with isotropic stiffness.
  Params are Young's modulus and Poisson's ratio.
  """
  (c,E,n) = params
  G = E/(2*(1+n))
  return fung_P(F,c,el.orthotropic_stiffness(E,E,E,n,n,n,G,G,G))

def model_isotropic_D(F,*params):
  """
  Exposed fung_D with isotropic stiffness.
  Params are Young's modulus and Poisson's ratio.
  """
  (c,E,n) = params
  G = E/(2*(1+n))
  return fung_D(F,c,el.orthotropic_stiffness(E,E,E,n,n,n,G,G,G))


def fung(F,c,CC):
  """
  Returns the strain energy given the deformation, ``F``, and the model
  parameters.
  W = c/2 (exp(Q) - 1)
  Q = E:CC:E
  """
  dim = F.shape[-1]
  if len(F.shape)>2:
    I = np.tile(np.eye(dim),(F.shape[0],1,1))
    CC = np.tile(CC,(F.shape[0],1,1,1,1))
  else:
    I = np.eye(dim)
  E = 0.5*np.einsum('...ab,...ac',F,F) - 0.5*I
  Q = np.einsum('...abcd,...abcd',CC,np.einsum('...ab,...cd',E,E))
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
  Q = np.einsum('...abcd,...abcd',CC,np.einsum('...ab,...cd',E,E))

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
  if len(F.shape)>2:
    I = np.tile(np.eye(dimension),(F.shape[0],1,1))
    CC = np.tile(CC,(F.shape[0],1,1,1,1))
  else:
    I = np.eye(dimension)
  E = 0.5*np.einsum('...ab,...ac',F,F) - 0.5*I
  Q = np.einsum('...abcd,...abcd',CC,np.einsum('...ab,...cd',E,E))

  # Derivative of Q wrt E: (rank 2)
  dQdE = 2*np.einsum('...abcd,...cd',CC,E)
  # Derivative of E wrt F: (rank 4)
  dEdF = 0.5*(np.einsum('...ab,...cd->...adcb',I,F) + np.einsum('...ab,...cd->...bdca',I,F))
  # Derivative of Q wrt F: (dQdE)ij (dEdF)ijkl (rank 2)
  dQdF = np.einsum('...ij,...ijkl',dQdE,dEdF)

  # Second derivative of Q wrt E: (rank 4)
  ddQdEdE = 2*CC
  # Second derivative of E wrt F: (rank 6)
  ddEdFdF = 0.5*(np.einsum('...ij,...kl,...mn->...kmilnj',I,I,I) + 
                 np.einsum('...ij,...kl,...mn->...kmjlni',I,I,I))

  # Each term of 3-part product rule
  ddW1 = lin.tensor(dQdF,dQdF)
  ddW2 = np.einsum('...ijkl,...ijmnklpq',ddQdEdE,np.einsum('...ijmn,...klpq',dEdF,dEdF))
  ddW3 = np.einsum('...ij,...ijklmn',dQdE,ddEdFdF)
  # Return the sum of these, all times c/2 exp(Q)
  return 0.5*c*(np.exp(Q)*(ddW1 + ddW2 + ddW3).T).T

