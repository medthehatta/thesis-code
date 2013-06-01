"""
fung_incompressible_invariants.py

Attempt to implement the invariant version of fung including incompressibility.
Hopefully it isn't too slow and annoying to call.
"""

import elastic as el
import lintools as lin
import numpy as np
from itertools import count


def Qbar(E,c,M,L,P=None):
    """
    Computes the distortional ``Q`` in the invariant Fung model.

    Required arguments:
     - Modified Lagrangian strain ``E``
     - Scalar parameter ``c``
     - List of 3 parameters ``M``
     - List of 6 "upper triangle" parameters ``L``

    Optional arguments:
     - Orthotropic structure tensor ``P``

    Returns:
     - Distortional ``Q`` (a scalar)
    """
    
    # Get orthotropic projection operators
    P = P or el.standard_orthotropic_P()
    A = el.orthotropic_projectors(P)

    # Compute scalar invariants
    EE = np.dot(E,E)
    EEA = np.tensordot(EE,A)
    EA = np.tensordot(E,A)

    # Express model with parameters
    # (2mi(Ai:E^2) + lij(Ai:E)(Aj:E))/c
    M_part = 2*np.dot(M,EEA)
    L_part = np.dot(L,lin.utri_flat(np.outer(EA,EA)))
    # TODO: For some reason this returns a singleton list instead of an actual
    # scalar
    return (M_part + L_part)/c


def Sbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or el.standard_orthotropic_P()


    




















def Cbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or el.standard_orthotropic_P()
    pass

def C(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or el.standard_orthotropic_P()
    pass

