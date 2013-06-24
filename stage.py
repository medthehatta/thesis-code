import numpy as np
import lintools as lin

def II(EA,EEA):
    OI = 0.5*(EEA-EA**2)

    I1 = np.zeros((3,3))
    for i in range(3):
        I1[i,i] = EA[i]**2
    
    if (OI!=0).any():
        I1[0,1] = np.sum([1,1,-1]*OI,axis=0)
        I1[1,0] = I1[0,1]
        I1[1,2] = np.sum([-1,1,1]*OI,axis=0)
        I1[2,1] = I1[1,2]
        I1[0,2] = np.sum([1,-1,1]*OI,axis=0)
        I1[2,0] = I1[0,2]

    return np.sqrt(I1)

def Q(EA,EEA,c,M,L):
    M_part = np.dot(M,EEA)
    L_part = np.tensordot(L,lin.tensor(EA,EA))
    return (2*M_part+L_part)/c

def S(EA,EEA,A,c,M,L):
    mII = II(EA,EEA)
    M_part = np.tensordot(M*mII,A) 
    L_part = np.sum([2*A[i,i]*L[i,:]*np.diag(mII) for i in range(3)])
    return np.exp(Q(EA,EEA,c,M,L))*(M_part+0.5*L_part)

