"""
fung_incompressible_invariants.py

Attempt to implement the invariant version of fung including incompressibility.
Hopefully it isn't too slow and annoying to call.
"""

import elastic as el
import lintools as lin
import numpy as np
from itertools import count

import pdb

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
        EEA = np.tensordot(np.dot(E,E),A,axes=([0,1],[1,2]))

    if EA is None:
        EA = np.tensordot(E,A,axes=([0,1],[1,2]))

    # Express model with parameters
    # (2mi(Ai:E^2) + lij(Ai:E)(Aj:E))/c
    M_part = 2*np.dot(M,EEA)
    L_part = np.dot(L,lin.utri_flat(np.outer(EA,EA)))
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
        EEA = np.tensordot(np.dot(E,E),A,axes=([0,1],[1,2]))

    if EA is None:
        EA = np.tensordot(E,A,axes=([0,1],[1,2]))

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
        EEA = np.tensordot(np.dot(E,E),A,axes=([0,1],[1,2]))

    if EA is None:
        EA = np.tensordot(E,A,axes=([0,1],[1,2]))

    # Compute quantities {E.Ai}
    if EdA is None:
        # LAME: np.dot broadcasts differently than einsum, so we can't use it
        # for this operation, even though it's way faster.  Crappy.
        proper_dot = lambda x,y: np.einsum('...ab,...bc',x,y)
        EdA = lin.anticommutator(E,A,op=proper_dot)

    # Compute Q
    if Q is None:
        Q = Qbar(E,c,M,L,EA,EEA,A=A)

    # Compute S
    if S is None:
        S = Sbar(E,c,M,L,EA,EEA,EdA,Q,A)

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

    return 2*np.exp(-Q)*SxS/c + np.exp(Q)*(M_part + L_part)




def tangent_stiffness(E,p,dpdJ,c,M,L,J=1,EA=None,EEA=None,EdA=None,Q=None,A=None,\
      S=None,AxA=None,AsI=None,CC=None,C=None,Ci=None,\
      FTF=None,FTFi=None):
    """
    Computes the tangent stiffness for the invariant Fung model.

    Parameters
    ----------
    E : (3, 3) array_like
        Modified Lagrangian strain.
    p : float
        Pressure
    dpdJ : float
        Pressure "modulus" (dp/dJ)
    c : float
        Scalar parameter.
    M : (3,) array_like
        List of 3 parameters.
    L : (6,) array_like
        List of 6 "upper triangle" parameters.

    Optional Parameters
    -------------------
    J : float
        Determinant of the deformation gradient (default: 1)
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
    CC : (3, 3, 3, 3) array_like
        Distortional tangent stiffness
    C : (3, 3) array_like
        Distortional Right Cauchy-Green tensor
    Ci : (3, 3) array_like
        Inverse of C
    FTF : (3, 3) array_like
        Actual Right Cauchy-Green tensor
    FTFi : (3, 3) array_like
        Inverse of FTF

    Returns
    -------
    output : (3, 3, 3, 3) array_like
        Tangent stiffness
    """
    
    # Get orthotropic projection operators
    if A is None:
        A = el.orthotropic_projectors(el.standard_orthotropic_P())

    # Compute scalar invariants
    if EEA is None:
        EEA = np.tensordot(np.dot(E,E),A,axes=([0,1],[1,2]))

    if EA is None:
        EA = np.tensordot(E,A,axes=([0,1],[1,2]))

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
        S = Sbar(E,c,M,L,EA,EEA,EdA,Q,A)

    # Compute S(x)S
    SxS = lin.tensor(S,S)

    # Compute {Ai(x)Aj}
    if AxA is None:
        AxA = np.array([lin.anticommutator(A[i],A[j],op=lin.tensor)
                        for (i,j) in lin.utri_indices(3)])

    # Compute {Ai(sym)I}
    if AsI is None:
        AsI = lin.anticommutator(A,np.eye(3),op=lin.symmetric_kronecker)

    # Compute CC (distortional C)
    if CC is None:
        CC = Cbar(E,c,M,L,EA,EEA,EdA,Q,A,S,AxA,AsI)

    # Compute C (distortional right cauchy-green)
    if C is None:
        C = 2*E + np.eye(3)

    # Compute C inverse
    if Ci is None:
        Ci = np.linalg.inv(C)

    # Compute FTF
    if FTF is None:
        FTF = J**(2/3.)*C

    # Compute FTF inverse
    if FTFi is None:
        FTFi = J**(2/3.)*Ci #  TODO: check

    # Compute various products of the FTF's
    FTFisFTFi = lin.symmetric_kronecker(FTFi,FTFi)
    FTFixFTFi = lin.tensor(FTFi,FTFi)

    # Compute various products of the deviatoric C
    CxCi = lin.tensor(C,Ci)
    CixC = lin.tensor(Ci,C)
    CisCi = lin.symmetric_kronecker(Ci,Ci)
    CxC = lin.tensor(C,C)
    CixCi = lin.tensor(Ci,Ci)
    SCi = lin.anticommutator(S,Ci,op=lin.tensor)
    CS = np.tensordot(C,S)

    # Compute various products with the deviatoric tangent stiffness CC
    CCCC = np.tensordot(np.tensordot(CC,C),C)
    CCCCi = np.tensordot(CC,CxCi) + np.tensordot(CixC,CC)
    CS = np.tensordot(C,S)

    # Assemble expression
    part_A = 2*J*p*FTFisFTFi
    part_B = J*(p + J*dpdJ)*FTFixFTFi
    part_C = CCCCi - (1/3.)*CCCC*CixCi
    part_D = CisCi + (1/3.)*CixCi
    part_E = CS*part_D - SCi

    return part_A - part_B + J**(-4/3.)*(CC - (1/3.)*part_C + (2/3.)*part_E)


