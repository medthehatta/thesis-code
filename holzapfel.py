"""
holzapfel.py

Constitutive model and tangent stiffness for the Holzapfel aorta model.
"""
import numpy as np

def iso_material_model(F,H,*params):
    """
    Distortional PK2 stress vs F for the Holzapfel model.
    Assume J=1 (incompressibility).

    Parameters
    ----------
    F : deformation gradient
    A1, A2 : projection operators along fiber directions
    params : c, k1, k2

    Returns
    -------
    PK2 stress
    """
    
    # Extract parameters from parameter vector
    (A1,A2) = H
    (c,k1,k2) = params

    # Compute necessary basis tensors for the expression
    C = np.dot(F.T,F)
    I = np.eye(3)
    Ci = np.linalg.inv(C)

    # Alias the invariants of C
    I0 = np.trace(C)
    I1 = np.tensordot(C,A1)
    I2 = np.tensordot(C,A2)

    # Alias the quantity (Ibar - 1)
    # J doesn't appear because it's 1
    Q1 = I1 - 1  
    Q2 = I2 - 1

    # Collect the constituent terms in the stress
    isotropic_part = c*(I - (1/3.)*I0*Ci)
    fiber_part1 = k1*Q1*np.exp(k2*Q1*Q1)*(A1 - (1/3.)*I1*Ci)
    fiber_part2 = k2*Q2*np.exp(k2*Q2*Q2)*(A2 - (1/3.)*I2*Ci)

    return isotropic_part + fiber_part1 + fiber_part2


def iso_material_elasticity(F,H,*params):
    """
    Isochoric material elasticity tensor from F, structure tensors in
    H, and parameters.
    """
    # Extract parameters from the parameter vector
    (A1,A2) = H
    (c,k1,k2) = params

    # Compute the necessary basis tensors
    C = np.dot(F.T,F)
    I = np.eye(3)
    Ci = np.linalg.inv(C)

    # Define the index notation for the various products
    tensor = '...ab,...cd->abcd'
    kronecker = '...ac,...bd->abcd'
    cokronecker = '...ad,...bc->abcd'

    # Tensor products of the C's
    CixCi = np.einsum(tensor,Ci,Ci)
    CixI = np.einsum(tensor,Ci,I)
    IxCi = np.einsum(tensor,I,Ci)
    CivI = CixI + IxCi
    CipCi = 0.5*np.einsum(kronecker,Ci,Ci) + \
            0.5*(1/3.)*np.einsum(cokronecker,Ci,Ci)

    # Tensor products of the A's
    A1xA1 = np.einsum(tensor,A1,A1)
    A2xA2 = np.einsum(tensor,A2,A2)

    # Mixed products
    CixA1 = np.einsum(tensor,Ci,A1)
    A1xCi = np.einsum(tensor,A1,Ci)
    CivA1 = CixA1 + A1xCi
    CixA2 = np.einsum(tensor,Ci,A2)
    A2xCi = np.einsum(tensor,A2,Ci)
    CivA2 = CixA2 + A2xCi

    # Alias the anisotropic invariants
    I0 = np.trace(C)
    I1 = np.tensordot(C,A1)
    I2 = np.tensordot(C,A2)

    # Alias the quantity (Ibar - 1)
    # There is no J, because we assume J=1
    Q1 = I1 - 1
    Q2 = I2 - 1

    # Collect the constituent terms
    isotropic1 = (c/2)*(0.5*I0*CipCi - (1/3.)*CivI)
    distortional01 = k1*np.exp(k2*Q1*Q1)
    distortional02 = k1*np.exp(k2*Q2*Q2)
    distortional11 = Q1*(0.5*I1*CipCi - (1/3.)*CivA1)
    distortional12 = Q2*(0.5*I2*CipCi - (1/3.)*CivA2)
    distortional21 = (1+2*k2*Q1*Q1)*(A1xA1 + (1/9.)*I1*I1*CixCi - (1/3.)*I1*CivA1)
    distortional22 = (1+2*k2*Q2*Q2)*(A2xA2 + (1/9.)*I2*I2*CixCi - (2/3.)*I2*CivA2)

    return isotropic1 + \
           distortional01*(distortional11 + distortional21) + \
           distortional02*(distortional12 + distortional22)



def vol_spatial_model(F,p,H,*params):
    I = np.eye(3)
    return -p*I


def iso_spatial_model(F,H,*params):
    So = material_constitutive_model(F,H,*params)
    return np.dot(F,np.dot(S,F.T))


def vol_model(F,p,H,*params):
    Fi = np.linalg.inv(F)
    return -p*Fi


def iso_model(F,H,*params):
    So = material_constitutive_model(F,H,*params)
    return np.dot(F,So)


def vol_material_model(F,p,H,*params):
    C = np.dot(F.T,F)
    Ci = np.linalg.inv(C)
    return -p*Ci


def vol_material_elasticity(F,p,pt,H,*params):
    C = np.dot(F.T,F)
    Ci = np.linalg.inv(C)
    return pt*lin.tensor(Ci,Ci) - 2*p*lin.symmetric_kronecker(Ci,Ci)
