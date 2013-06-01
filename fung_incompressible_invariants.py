"""
fung_incompressible_invariants.py

Attempt to implement the invariant version of fung including incompressibility.
Hopefully it isn't too slow and annoying to call.
"""

import elastic as el
import lintools as lin
import numpy as np
from itertools import count


def Qbar(E,c,M,L,EA=None,EEA=None,A=None):
    """
    Computes the distortional ``Q`` in the invariant Fung model.

    Parameters
    ----------
    E : (3, 3) array_like
        Modified Lagrangian strain.
    c : float
        Scalar parameter.
    M : array (3,)
        List of 3 parameters.
    L : array (6,)
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    EA : array (3,)
        Pre-computed invariants E:Ai
    EEA : array (3,)
        Pre-computed invariants E^2:Ai
    A : array (3,3,3)
        Pre-computed projectors onto the planes of orthotropy 

    Returns
    -------
    output : array (1,)
        Distortional Q
    """

    # Get orthotropic projection operators
    if A is None:
        A = el.orthotropic_projectors(el.standard_orthotropic_P())

    # Compute scalar invariants
    if EEA is None:
        EEA = np.tensordot(np.dot(E,E),A)

    if EA is None:
        EA = np.tensordot(E,A)

    # Express model with parameters
    # (2mi(Ai:E^2) + lij(Ai:E)(Aj:E))/c
    M_part = 2*np.dot(M,EEA)
    L_part = np.dot(L,lin.utri_flat(np.outer(EA,EA)))
    # TODO: For some reason this returns a singleton list instead of an actual
    # scalar
    return (M_part + L_part)/c


def Sbar(E,c,M,L,EA=None,EdA=None,Q=None,A=None):
    """
    Computes the distortional ``S`` in the invariant Fung model.

    Parameters
    ----------
    E : (3, 3) array_like
        Modified Lagrangian strain.
    c : float
        Scalar parameter.
    M : array (3,)
        List of 3 parameters.
    L : array (6,)
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    EA : array (3,)
        Pre-computed invariants E:Ai
    EdA : array (3,3,3)
        Pre-computed quantities {E.Ai}
    Q : float
        Pre-computed Q
    A : array (3,3,3)
        Pre-computed projectors onto the planes of orthotropy 

    Returns
    -------
    output : (3, 3) array_like
        Distortional S (PK2 stress)
    """
    
    # Get orthotropic projection operators
    if A is None:
        A = el.orthotropic_projectors(el.standard_orthotropic_P())

    # Compute scalar invariants
    if EA is None:
        EA = np.tensordot(E,A)

    # Compute quantities {E.Ai}
    if EdA is None:
        EdA = lin.anticommutator(E,A)

    # Compute {(Ai(x)Aj) : E}
    AAE = np.array([A[i]*EA[j] + A[j]*EA[i] for (i,j) in lin.utri_indices(3)])

    # Compute Q
    if Q is None:
        Q = Qbar(E,c,M,L,EA,EEA=None,A=A)

    # Insert parameters
    M_part = sum(M*EdA)
    L_part = 0.5*sum((AAE.T*L).T) # 0.5 * Li * AAEi

    return np.exp(Q)*(M_part + L_part)
    

def Cbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or el.standard_orthotropic_P()
    pass

def C(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or el.standard_orthotropic_P()
    pass


