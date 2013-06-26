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
    BB = np.dot(B,B)
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

    # Compute the relevant products of B
    BxB = np.einsum(tensor,B,B)
    BsB = 0.5*(np.einsum(kronecker,B,B) + np.einsum(cokronecker,B,B))

    # Compute the mixed products of B, I, and BB
    BxI = np.einsum(tensor,B,I)
    IxB = np.einsum(tensor,I,B)
    BvI = BxI + IxB
    BBxI = np.einsum(tensor,BB,I)
    IxBB = np.einsum(tensor,I,BB)
    BBvI = BBxI + IxBB

    # Assemble the expression
    part1 = (1/3.)*(I1*(IsI + (1/3.)*IxI) - BvI)
    part2 = (2/3.)*(BBvI - I1*BvI + I2*(IsI + (1/3.)*IxI))
    part3 = (2/3.)*(BxB - (0.5)*BsB)

    return c1*part1 + c2*(part2 + part3)

def constitutive_model(B,*p):
    """
    Kirchhoff stress as a function of the left Cauchy-green tensor and the two
    parameters: p=c1,c2
    """
   
    # Extract the model parameters
    (c1,c2) = p

    # Compute the other required generating tensors for the expression
    I = np.eye(3)
    BB = np.dot(B,B)

    # Alias the invariants of B
    I1 = np.trace(B)
    I2 = 0.5*(I1*I1 - np.trace(np.dot(B,B)))

    # Assemble the expression
    dIb1dB = B - (1/3.)*I1*I
    dIb2dB = I1*B - (2/3.)*I2*I - (0.5)*BB
    dWdB = c1*dIb1dB + c2*dIb2dB

    # I appear to be off by a sign.  I'll have to fix this properly later
    return -2*dWdB



