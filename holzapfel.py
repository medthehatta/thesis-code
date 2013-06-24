"""
holzapfel.py

Constitutive model and tangent stiffness for the Holzapfel aorta model.
"""
import numpy as np

def constitutive_model(C,A1,A2,*params):
    """
    PK2 stress vs C for the Holzapfel model.
    Assume J=1 (incompressibility).

    TODO: Shouldn't there be pressure in here someplace?
          I guess this is the "distortional" stress.

    Parameters
    ----------
    C : right Cauchy-Green tensor
    A1, A2 : projection operators along fiber directions
    params : c, k1, k2

    Returns
    -------
    PK2 stress
    """
    
    # Extract parameters from parameter vector
    (c,k1,k2) = params

    # Compute necessary basis tensors for the expression
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


