#!/usr/bin/env python3
# coding: utf8
#
# mooney_rivlin.py
# Mooney Rivlin constitutive model and tangent stiffness
#

import numpy as np

def first_elasticity(F,pressure,*p):
    """
    d^2 W / dF^2
    """
    I = np.eye(3)
    D = material_tangent_stiffness(F,pressure,*p)
    E = 0.5*(np.dot(F.T,F) - I)
    S = material_constitutive_model(E,pressure,*p)

    return np.einsum('mjnl,kn,im->ijkl',D,F,F) + np.einsum('jl,ik->ijkl',S,I)


def material_tangent_stiffness(F,pressure,*p):
    """
    Material Tangent stiffness as a function of the deformation gradient and
    the two parameters: p=c1,c2
    """
    
    # Extract the model parameters
    (c10,c01) = p

    # Compute the other required generating tensors for the expression
    C = np.dot(F.T,F)
    Ci = np.linalg.inv(C)
    I = np.eye(3)

    # Alias the scalar invariants of C
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    # Alias the index manipulations for the various tensor products
    tensor = '...ab,...cd->abcd'
    kronecker = '...ac,...bd->abcd'
    cokronecker = '...ad,...bc->abcd'

    # Compute the relevant products of I
    IxI = np.einsum(tensor,I,I)
    IsI = 0.5*(np.einsum(kronecker,I,I) + np.einsum(cokronecker,I,I))

    # Compute the relevant products of C and Ci
    CixCi = np.einsum(tensor,Ci,Ci)
    CixC = np.einsum(tensor,Ci,C)
    CxCi = np.einsum(tensor,C,Ci)
    CvCi = CxCi + CixC
    CisCi = 0.5*(np.einsum(kronecker,Ci,Ci) + np.einsum(cokronecker,Ci,Ci))

    # Compute the mixed products of C, I, and Ci
    CixI = np.einsum(tensor,Ci,I)
    IxCi = np.einsum(tensor,I,Ci)
    CivI = CixI + IxCi
    CxI = np.einsum(tensor,C,I)
    IxC = np.einsum(tensor,I,C)
    CvI = CxI + IxC

    # Assemble the expression
    part0 = CixCi - 2*CisCi  # 4*(1/2)*( (1/2)CixCi - CisCi )
    part1 = I1*(CisCi + (1/3.)*CixCi) - CivI

    part21 = CvCi - 2*I1*CivI + I2*(CisCi + (2/3.)*CixCi)
    part22 = 2*IxI - IsI
    part2 = (2/3.)*part21 + part22

    return pressure*part0 + 4*(c10*(1/3.)*part1 + c01*part2)


def spatial_constitutive_model(b,pressure,*p):
    """
    Cauchy stress as a function of the deformation gradient and model parameters.
    Assume incompressibility.  (So this is also Kirchhoff stress.)
    """

    # Extract the model parameters
    (c10,c01) = p

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    bb = np.dot(b,b)

    # Alias the invariants of b
    I1 = np.trace(b)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(b,b)))

    # Assemble the expression
    volumetric = -pressure*I
    part1 = c10*b
    part2 = c01*(2*I1*b - bb)

    return volumetric + 2*(part1 + part2)


def constitutive_model(F,pressure,*p):
    """
    PK1 stress as a function of the deformation gradient and model parameters.
    """
   
    # Extract the model parameters
    (c10,c01) = p

    # Compute the other required generating tensors for the expression
    C = np.dot(F.T,F)
    Fit = np.linalg.inv(F.T)
    FC = np.dot(F,C)

    # Alias the invariants of C
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    # Assemble the expression
    volumetric = -pressure*Fit
    part1 = c10*F
    part2 = c01*(2*I1*F - FC)

    return volumetric + 2*(part1 + part2)


def material_constitutive_model(E,pressure,*p):
    """
    PK2 stress as a function of the Lagrangian strain and model parameters.
    """
   
    # Extract the model parameters
    (c10,c01) = p

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    C = 2*E + I
    Ci = np.linalg.inv(C)

    # Alias the invariants of C
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    # Assemble the expression
    volumetric = -pressure*Ci
    part1 = c10*I
    part2 = c01*(2*I1*I - C)

    return volumetric + 2*(part1 + part2)


def strain_energy_density(F,*p):
    """
    Strain energy density.
    """
    
    # Extract model parameters
    (c10,c01) = p

    # Right Cauchy-Green
    C = np.dot(F.T,F)

    # Compute invariants
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    return c10*(I1 - 3) + c01*(I2 - 3)

