#!/usr/bin/env python3
# coding: utf8
#
# mooney_rivlin.py
# Mooney Rivlin constitutive model and tangent stiffness
#

import numpy as np

def tangent_stiffness(B,*p):
    """
    Tangent stiffness as a function of the left Cauchy-green tensor and the two
    parameters: p=c1,c2
    """
    
    # Extract the model parameters
    (c1,c2) = p

    # Compute the other required generating tensors for the expression
    Bi = np.linalg.inv(B)
    I = np.eye(3)

    # Alias the scalar invariants of B
    I1 = np.trace(B)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(B,B)))

    # Alias the index manipulations for the various tensor products
    tensor = '...ab,...cd->abcd'
    kronecker = '...ac,...bd->abcd'
    cokronecker = '...ad,...cb->abcd'

    # Compute the relevant products of I
    IxI = np.einsum(tensor,I,I)
    IsI = 0.5*(np.einsum(kronecker,I,I) + np.einsum(cokronecker,I,I))

    # Compute the relevant products of B and Bi
    BixBi = np.einsum(tensor,Bi,Bi)
    BixB = np.einsum(tensor,Bi,B)
    BxBi = np.einsum(tensor,B,Bi)
    BvBi = BxBi + BixB
    BisBi = 0.5*(np.einsum(kronecker,Bi,Bi) + np.einsum(cokronecker,Bi,Bi))

    # Compute the mixed products of B, I, and Bi
    BixI = np.einsum(tensor,Bi,I)
    IxBi = np.einsum(tensor,I,Bi)
    BivI = BixI + IxBi

    # Assemble the expression
    third = (1/3.) #  this is probably unnecessary, but hey.
    part1 = I1*(BisBi + third*BixBi) - BivI
    part21 = BvBi - I1*BivI + I2*(BisBi + third*BixBi)
    part22 = IxI - 0.5*IsI
    d2Wdb2 = c1*third*part1 + c2*2*third*part21 + c2*part22

    return 4*np.einsum('...ab,...bcde,...ef',B,d2Wdb2,B)

def constitutive_model(B,*p):
    """
    Kirchhoff stress as a function of the left Cauchy-green tensor and the two
    parameters: p=c1,c2
    """
   
    # Extract the model parameters
    (c1,c2) = p

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    Bi = np.linalg.inv(B)

    # Alias the invariants of B
    I1 = np.trace(B)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(B,B)))

    # Assemble the expression
    dIb1dB = I - (1/3.)*I1*Bi
    dIb2dB = I1*I - (2/3.)*I2*Bi - (0.5)*B
    dWdB = c1*dIb1dB + c2*dIb2dB

    return 2*np.einsum('...ab,...bcde',B,dWdB)



