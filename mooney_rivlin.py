#!/usr/bin/env python3
# coding: utf8
#
# mooney_rivlin.py
# Mooney Rivlin constitutive model and tangent stiffness
#

import numpy as np

def material_tangent_stiffness(C,pressure,*p):
    """
    Material Tangent stiffness as a function of the right Cauchy-green tensor
    and the two parameters: p=c1,c2
    """
    
    # Extract the model parameters
    if len(p)==2:
        (c1,c2) = p
        c3 = 0
    else:
        (c1,c2,c3) = p

    # Compute the other required generating tensors for the expression
    Ci = np.linalg.inv(C)
    I = np.eye(3)

    # Alias the scalar invariants of C
    I1 = np.trace(C)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(C,C)))

    # Alias the index manipulations for the various tensor products
    tensor = '...ab,...cd->abcd'
    kronecker = '...ac,...bd->abcd'
    cokronecker = '...ad,...cb->abcd'

    # Compute the relevant products of I
    IxI = np.einsum(tensor,I,I)
    IvI = 2*IxI
    IsI = 0.5*(np.einsum(kronecker,I,I) + np.einsum(cokronecker,I,I))

    # Compute the relevant products of C and Ci
    CixCi = np.einsum(tensor,Ci,Ci)
    CivCi = 2*CixCi
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
    part0 = CixCi - 2*CisCi
    part1 = I1*(CisCi + (1/3.)*CixCi) - CivI

    part21 = CvCi - I1*CivI + I2*(CisCi + (2/3.)*CixCi)
    part22 = IxI - IsI
    part2 = (2/3.)*part21 + part22

    return pressure*part0 + 4*(c1*(1/3.)*part1 + c2*part2)


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
    part2 = c01*(I1*F - FC)

    return volumetric + 2*(part1 + part2)



