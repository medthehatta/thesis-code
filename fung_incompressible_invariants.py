
def standard_orthotropic_P():
    e1 = [1,0,0]
    e2 = [0,1,0]
    P = np.outer(e1,e1) - np.outer(e2,e2)



def Qbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or standard_orthotropic_P()
    
    I = np.eye(3)

    M = [m1,m2,m3]
    L = [l11,l12,l13,l22,l23,l33]
                
    PP = np.dot(P,P)
    A1 = 0.5*(P + PP)
    A2 = 0.5*(P - PP)
    A3 = I - A1 - A2
    A = [A1,A2,A3]
       
    EE = np.dot(E,E)
    invariants_EE = [np.trace(np.dot(EE,M[i]*A[i])) for i in range(3)]

    invariants_E = [np.trace(np.dot(E,A[i]))*np.trace(np.dot(E,A[j])) 
                    for (i,j) in lin.utri_indices(3)]

    return (2*sum(invariants_EE) + sum(invariants_E))/c


def Sbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or standard_orthotropic_P()
    
    



def Cbar(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or standard_orthotropic_P()
    pass

def C(E,c,m1,m2,m3,l11,l12,l13,l22,l23,l33,P=None):
    P = P or standard_orthotropic_P()
    pass

