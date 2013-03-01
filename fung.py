"""
fung.py

Strain energy density and tangent stiffness for the Fung tissue model.
"""
import lintools as lin
import numpy as np
import pdb

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
  I = np.eye(dimension)
  E = 0.5*np.dot(F.T,F) - 0.5*I
  Q = np.einsum('abcd,ab,cd',CC,E,E)

  # Derivative of Q wrt E: (rank 2)
  dQdE = 2*np.einsum('abcd,cd',CC,E)
  # Derivative of E wrt F: (rank 4)
  dEdF = 0.5*np.einsum('ab,cd->adcb',I,F) + 0.5*np.einsum('ab,cd->bdca',I,F)
  # Derivative of Q wrt F: (dQdE)ij (dEdF)ijkl (rank 2)
  dQdF = np.einsum('ij,ijkl',dQdE,dEdF)
  # Return value: c/2 exp(Q) dQdF
  return 0.5*c*np.exp(Q)*dQdF

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

