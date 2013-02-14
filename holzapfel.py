import numpy as np
import pdb

def holzapfel(F,c,k1,k2,A1=np.outer([1,0,0],[1,0,0]),A2=np.outer([0,1,0],[0,1,0])):
  """
  Returns the strain energy given the deformation, ``F``, and the model
  parameters.
  A1 and A2 are the structure tensors for the fibers.
  """
  J  = np.linalg.det(F)
  I1 = J**(-2/3.) * np.trace(np.dot(F.T,F))
  I4 = J**(-2/3.) * np.einsum('ji,ik,kj',F.T,F,A1)
  I6 = J**(-2/3.) * np.einsum('ji,ik,kj',F.T,F,A2)

  W1 = c*(I1-3)/2.
  W2 = k1/(2*k2) * (np.exp(k2*(I4-1)*(I4-1))-1)
  W3 = k1/(2*k2) * (np.exp(k2*(I6-1)*(I6-1))-1)

  return W1+W2+W3

def holzapfel_D(F,c,k1,k2,A1=np.outer([1,0,0],[1,0,0]),A2=np.outer([0,1,0],[0,1,0])):
  """
  Returns the tangent stiffness given the deformation, ``F``, and the model
  parameters.
  A1 and A2 are the structure tensors for the fibers.
  """
  # simple invariants of F, or joint invariants of F and a structure tensor
  J  = np.linalg.det(F)
  J23 = np.power(J,-2/3.)
  trFF   = np.einsum('ij,ij',F,F)
  trFFA1 = np.einsum('ji,jk,ki',F,F,A1)
  trFFA2 = np.einsum('ji,jk,ki',F,F,A2)
  I1 = J23*trFF
  I4 = J23*trFFA1
  I6 = J23*trFFA2

  # tensors involving inverses of F
  Fi = np.linalg.inv(F)
  dFiTdF = -np.einsum('ij,kl->lijk',Fi,Fi) 
  FxFi = np.einsum('ij,kl->ijkl',F,Fi)

  # some identity tensors
  dim = len(F)
  I   = np.eye(dim)
  II  = np.einsum('ij,kl->ikjl',I,I)

  # parts of Gamma for the first set of fibers
  a4  = I4 - 1
  b4  = np.exp(k2*(I4-1)*(I4-1))
  g14 = trFFA1
  g24 = Fi.T
  g4  = np.dot(F,A1) - g14*g24/3.
  # derivatives of parts of Gamma for the first set of fibers
  da4 = 2*J23*(np.dot(F,A1) - trFFA1*Fi.T)
  db4 = 2*k2*(I4-1)*da4
  dg04 = np.einsum('ik,mn->imnk',I,A1)
  dg14 = 2*np.dot(F,A1)
  dg24 = dFiTdF
  dg4  = dg04 - np.einsum('ik,mn->ikmn',g24,dg14)/3. - g14*dg24/3.

  # parts of Gamma for the second set of fibers
  a6  = I6 - 1 
  b6  = np.exp(k2*(I6-1)*(I6-1)) 
  g16 = trFFA2  
  g26 = Fi.T 
  g6  = np.dot(F,A2) - g16*g26/3. 
  # derivatives of parts of Gamma for the second set of fibers
  da6 = 2*J23*(np.dot(F,A2) - trFFA2*Fi.T) 
  db6 = 2*k2*(I6-1)*da6 
  dg06 = np.einsum('ik,mn->imnk',I,A2)
  dg16 = 2*np.dot(F,A2) 
  dg26 = dFiTdF 
  dg6  = dg06 - np.einsum('ik,mn->ikmn',g26,dg16)/3. - g16*dg26/3.

  # Gammas
  G0 = c*F - c*trFF/3.*Fi.T
  G1 = a4*b4*g4
  G2 = a6*b6*g6
  # derivatives of Gammas
  dG0 = c*II - c/3.*(2*FxFi + trFF*dFiTdF)
  dG1 = b4*np.einsum('ij,kl->ijkl',da4,g4) + a4*np.einsum('ij,kl->ijkl',db4,g4) + a4*b4*dg4
  dG2 = b6*np.einsum('ij,kl->ijkl',da6,g6) + a6*np.einsum('ij,kl->ijkl',db6,g6) + a6*b6*dg6

  # the actual tangent stiffness is in two parts
  D10 = -2/3.*J23*Fi.T
  D11 = (G0 + 2*k1*G1 + 2*k1*G2)
  D1  = np.einsum('ij,kl->ijkl',D10,D11)
  D2  = J23*(dG0 + 2*k1*dG1 + 2*k1*dG2)

  return D1 + D2

