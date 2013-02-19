import numpy as np
import pdb

def tensor(A,B):
  """
  Return the tensor product of matrices A and B
  """
  return np.einsum('ij,kl->ijkl',A,B)

def holzapfel(F,c,k1,k2,A1=np.outer([1,0,0],[1,0,0]),A2=np.outer([0,1,0],[0,1,0])):
  """
  Returns the strain energy given the deformation, ``F``, and the model
  parameters.
  A1 and A2 are the structure tensors for the fibers.
  """
  J  = np.linalg.det(F)
  I1 = np.power(J,-2/3.) * np.trace(np.dot(F.T,F))
  I4 = np.power(J,-2/3.) * np.einsum('ji,ik,kj',F.T,F,A1)
  I6 = np.power(J,-2/3.) * np.einsum('ji,ik,kj',F.T,F,A2)

  W1 = c*(I1-3)/2.
  W2 = k1/(2*k2) * (np.exp(k2*(I4-1)*(I4-1))-1)
  W3 = k1/(2*k2) * (np.exp(k2*(I6-1)*(I6-1))-1)

  return W1+W2+W3

def holzapfel_D(F,c,k1,k2,A1=np.outer([1,0,0],[1,0,0]),A2=np.outer([0,1,0],[0,1,0])):
  """
  Returns the tangent stiffness given the deformation, ``F``, and the model
  parameters.
  A1 and A2 are the structure tensors for the fibers.

  Description of convenience variables:
  W = psi0 + psi1 + psi2
  psi0 = c/2 (I0-3)
  psii = k1/(2k2)Ei
  Ei = exp(k2Hi^2)
  Hi = Ii-1
  I0 = GS
  Ii = GSi
  G = det(F)^(-2/3)
  S = tr(F^TF)
  Si = tr(F^TFAi)
  """
  # simple invariants of F, or joint invariants of F and a structure tensor
  J  = np.linalg.det(F)
  G = np.power(J,-2/3.)
  S = np.einsum('ij,ij',F,F)
  S1 = np.einsum('ji,jk,ki',F,F,A1)
  S2 = np.einsum('ji,jk,ki',F,F,A2)
  I0 = G*S
  I1 = G*S1
  I2 = G*S2

  # inverse transpose of F
  Finv = np.linalg.inv(F)
  FiT  = Finv.T

  # some identity tensors
  dim = len(F)
  I   = np.eye(dim)
  II  = np.einsum('ij,kl->ikjl',I,I)

  # H and E
  H1 = I1-1
  H2 = I2-1
  E1 = np.exp(k2*H1*H1)
  E2 = np.exp(k2*H2*H2)

  # first derivatives
  dG = -2/3.*G*FiT
  dS = 2*F
  dS1 = 2*np.dot(F,A1)
  dS2 = 2*np.dot(F,A2)
  dI0 = S*dG + G*dS
  dI1 = S1*dG + G*dS1
  dI2 = S2*dG + G*dS2
  dF  = II
  dFiT = np.einsum('ij,kl->lijk',Finv,Finv)

  # second derivatives
  ddG = -2/3.*G*dFiT - 4/9.*G*tensor(FiT,FiT)
  ddS = 2*dF
  ddS1 = 2*np.einsum('ij,jklm',A1,dF) #2*A1.dF
  ddS2 = 2*np.einsum('ij,jklm',A2,dF) #2*A2.dF
  ddI0 = S*ddG + tensor(dG,dS) + tensor(dS,dG) + G*ddS
  ddI1 = S1*ddG + tensor(dG,dS1) + tensor(dS1,dG) + G*ddS1
  ddI2 = S2*ddG + tensor(dG,dS2) + tensor(dS2,dG) + G*ddS2

  # second derivatives of the psis
  ddpsi0 = c/2.*ddI0
  ddpsi1 = k1*E1*((2*k2*H1*H1+1)*tensor(dI1,dI1) + H1*ddI1)
  ddpsi2 = k1*E2*((2*k2*H2*H2+1)*tensor(dI2,dI2) + H2*ddI2)

  return ddpsi0 + ddpsi1 + ddpsi2


