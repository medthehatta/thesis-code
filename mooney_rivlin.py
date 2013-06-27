#!/usr/bin/env python3
# coding: utf8
#
# mooney_rivlin.py
# Mooney Rivlin constitutive model and tangent stiffness
#

import numpy as np

def spatial_tangent_stiffness(b,*p):
    """
    Spatial Tangent stiffness as a function of the left Cauchy-green tensor and
    the two parameters: p=c1,c2
    """
    
    # Extract the model parameters
    (c1,c2) = p

    # Compute the other required generating tensors for the expression
    bb = np.dot(b,b)
    I = np.eye(3)

    # Alias the scalar invariants of b
    I1 = np.trace(b)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(b,b)))

    # Alias the index manipulations for the various tensor products
    tensor = '...ab,...cd->abcd'
    kronecker = '...ac,...bd->abcd'
    cokronecker = '...ad,...cb->abcd'

    # Compute the relevant products of I
    IxI = np.einsum(tensor,I,I)
    IsI = 0.5*(np.einsum(kronecker,I,I) + np.einsum(cokronecker,I,I))

    # Compute the relevant products of b
    bxb = np.einsum(tensor,b,b)
    bsb = 0.5*(np.einsum(kronecker,b,b) + np.einsum(cokronecker,b,b))

    # Compute the mixed products of b, I, and bb
    bxI = np.einsum(tensor,b,I)
    Ixb = np.einsum(tensor,I,b)
    bvI = bxI + Ixb
    bbxI = np.einsum(tensor,bb,I)
    Ixbb = np.einsum(tensor,I,bb)
    bbvI = bbxI + Ixbb

    # Assemble the expression
    part1 = (1/3.)*(I1*(IsI + (1/3.)*IxI) - bvI)
    part2 = (2/3.)*(bbvI - I1*bvI + I2*(IsI + (1/3.)*IxI))
    part3 = (2/3.)*(bxb - (0.5)*bsb)

    return c1*part1 + c2*(part2 + part3)


def material_tangent_stiffness(C,*p):
    """
    Material Tangent stiffness as a function of the right Cauchy-green tensor
    and the two parameters: p=c1,c2
    """
    
    # Extract the model parameters
    (c1,c2) = p

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

    # Assemble the expression
    third = (1/3.) #  this is probably unnecessary, but hey.
    part1 = I1*(CisCi + third*CixCi) - CivI
    part21 = CvCi - I1*CivI + I2*(CisCi + third*CixCi)
    part22 = IxI - 0.5*IsI

    return c1*third*part1 + c2*2*third*part21 + c2*part22


def constitutive_model(b,*p):
    """
    Kirchhoff stress as a function of the left Cauchy-green tensor and the two
    parameters: p=c1,c2
    """
   
    # Extract the model parameters
    (c1,c2) = p

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    bb = np.dot(b,b)

    # Alias the invariants of b
    I1 = np.trace(b)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(b,b)))

    # Assemble the expression
    dIb1db = b - (1/3.)*I1*I
    dIb2db = I1*b - (2/3.)*I2*I - (0.5)*bb
    dWdb = c1*dIb1db + c2*dIb2db

    return 2*dWdb



