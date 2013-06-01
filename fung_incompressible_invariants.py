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
    M : (3,) array_like
        List of 3 parameters.
    L : (6,) array_like
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    EA : (3,) array_like
        Pre-computed invariants E:Ai
    EEA : (3,) array_like
        Pre-computed invariants E^2:Ai
    A : (3, 3, 3) array_like
        Pre-computed projectors onto the planes of orthotropy 

    Returns
    -------
    output : (1,) array_like
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


def Sbar(E,c,M,L,EA=None,EEA=None,EdA=None,Q=None,A=None):
    """
    Computes the distortional ``S`` in the invariant Fung model.

    Parameters
    ----------
    E : (3, 3) array_like
        Modified Lagrangian strain.
    c : float
        Scalar parameter.
    M : (3,) array_like
        List of 3 parameters.
    L : (6,) array_like
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    EA : (3,) array_like
        Pre-computed invariants E:Ai
    EEA : (3,) array_like
        Pre-computed invariants E^2:Ai
    EdA : (3, 3, 3) array_like
        Pre-computed quantities {E.Ai}
    Q : float
        Pre-computed Q
    A : (3, 3, 3) array_like
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
    if EEA is None:
        EEA = np.tensordot(np.dot(E,E),A)

    if EA is None:
        EA = np.tensordot(E,A)

    # Compute quantities {E.Ai}
    if EdA is None:
        # LAME: np.dot broadcasts differently than einsum, so we can't use it
        # for this operation, even though it's way faster.  Crappy.
        proper_dot = lambda x,y: np.einsum('...ab,...bc',x,y)
        EdA = lin.anticommutator(E,A,op=proper_dot)

    # Compute {(Ai(x)Aj) : E}
    AAE = np.array([A[i]*EA[j] + A[j]*EA[i] for (i,j) in lin.utri_indices(3)])

    # Compute Q
    if Q is None:
        Q = Qbar(E,c,M,L,EA,EEA=None,A=A)

    # Insert parameters
    M_part = sum((EdA.T*M).T)
    L_part = 0.5*sum((AAE.T*L).T) # 0.5 * Li * AAEi

    return np.exp(Q)*(M_part + L_part)
    


def Cbar(E,c,M,L,EA=None,EEA=None,EdA=None,Q=None,A=None,\
         S=None,AxA=None,AsI=None):
    """
    Computes the distortional ``C`` in the invariant Fung model.

    Parameters
    ----------
    E : (3, 3) array_like
        Modified Lagrangian strain.
    c : float
        Scalar parameter.
    M : (3,) array_like
        List of 3 parameters.
    L : (6,) array_like
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    EA : (3,) array_like
        Pre-computed invariants E:Ai
    EEA : (3,) array_like
        Pre-computed invariants E^2:Ai
    EdA : (3, 3, 3) array_like
        Pre-computed quantities {E.Ai}
    Q : float
        Pre-computed Q
    A : (3, 3, 3) array_like
        Pre-computed projectors onto the planes of orthotropy 
    S : (3, 3) array_like
        Pre-computed S
    AxA : (6, 3, 3, 3, 3) array_like
        Pre-computed list of A (tensor) A
    AsI : (6, 3, 3, 3, 3) array_like
        Pre-computed list of A (sym) I
        

    Returns
    -------
    output : (3, 3, 3, 3) array_like
        Distortional C (tangent stiffness)
    """
    
    # Get orthotropic projection operators
    if A is None:
        A = el.orthotropic_projectors(el.standard_orthotropic_P())

    # Compute scalar invariants
    if EEA is None:
        EEA = np.tensordot(np.dot(E,E),A)

    if EA is None:
        EA = np.tensordot(E,A)

    # Compute quantities {E.Ai}
    if EdA is None:
        # LAME: np.dot broadcasts differently than einsum, so we can't use it
        # for this operation, even though it's way faster.  Crappy.
        proper_dot = lambda x,y: np.einsum('...ab,...bc',x,y)
        EdA = lin.anticommutator(E,A,op=proper_dot)

    # Compute {(Ai(x)Aj) : E}
    AAE = np.array([A[i]*EA[j] + A[j]*EA[i] for (i,j) in lin.utri_indices(3)])

    # Compute Q
    if Q is None:
        Q = Qbar(E,c,M,L,EA,EEA,A=A)

    # Compute S
    if S is None:
        S = Sbar(E,c,M,L,EA,EdA,Q,A)

    # Compute S(x)S
    SxS = lin.tensor(S,S)

    # Compute {Ai(x)Aj}
    if AxA is None:
        AxA = np.array([lin.anticommutator(A[i],A[j],op=lin.tensor)
                        for (i,j) in lin.utri_indices(3)])

    # Compute {Ai(sym)I}
    if AsI is None:
        AsI = lin.anticommutator(A,np.eye(3),op=lin.symmetric_kronecker)

    # Insert parameters
    M_part = sum((AsI.T*M).T)
    L_part = 0.5*sum((AxA.T*L).T)
    S_part = 2*np.exp(-Q)*SxS/c

    return S_part + np.exp(Q)*(M_part + L_part)





