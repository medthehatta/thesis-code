#!/usr/bin/env python3
# coding: utf8
#
# mooney_rivlin.py
# Mooney Rivlin constitutive model and tangent stiffness
#

import numpy as np
import lintools as lin

def iso_material_elasticity(F,H,*params):
    """
    Isochoric Material elasticity tensor as a function of the deformation
    gradient and the two parameters: p=c1,c2
    """
    
    # Extract the model parameters
    (c10,c01) = params

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
    part1 = I1*(CisCi + (1/3.)*CixCi) - CivI

    part21 = CvCi - 2*I1*CivI + I2*(CisCi + (2/3.)*CixCi)
    part22 = 2*IxI - IsI
    part2 = (2/3.)*part21 + part22

    return 4*(c10*(1/3.)*part1 + c01*part2)


def iso_material_model(F,H,*params):
    """
    Isochoric PK2 stress as a function of the deformation gradient and model
    parameters.
    """
    # Extract the model parameters
    (c10,c01) = params

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    C = np.dot(F.T,F)
    Ci = np.linalg.inv(C)

    # Alias the invariants of C
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    # Assemble the expression
    part1 = 2*c10*I
    part2 = 2*c01*(2*I1*I - C)

    return part1 + part2


def strain_energy_density(F,H,*params):
    """
    Strain energy density.
    """
    # Extract model parameters
    (c10,c01) = params

    # Right Cauchy-Green
    C = np.dot(F.T,F)

    # Compute invariants
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    return c10*(I1 - 3) + c01*(I2 - 3)


def vol_material_elasticity(F,p,pt,H,*params):
    C = np.dot(F.T,F)
    Ci = np.linalg.inv(C)
    return pt*lin.tensor(Ci,Ci) - 2*p*lin.symmetric_kronecker(Ci,Ci)
