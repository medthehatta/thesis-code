import numpy as np
from scipy.linalg import expm
import lintools as lin
import elastic as el

import mooney_rivlin as mr

def tstiff(F, *p):
    pressure = el.pressure_PK1(F, mr.constitutive_model, *p)
    return mr.material_tangent_stiffness(F, pressure, *p)

def voigt_vec_to_mat(d):
    return np.array([[d[0],d[5],d[4]],
                     [d[5],d[1],d[3]],
                     [d[4],d[3],d[2]]]) 


dt = 0.01
F = np.eye(3)
Fs = [F]
for j in range(1000):
    # Compute D
    D = el.voigt(tstiff(F,*params))

    # Find a positive eigendirection
    (eigvals, eigvecs) = np.linalg.eigh(D)
    eigsystem = zip(eigvals,eigvecs)
    candidates = [vec for (val,vec) in eigsystem if val>0]
    if len(candidates)==0: break
    d = candidates[0]

    # Convert to matrix
    mat = voigt_vec_to_mat(d)

    # Exponentiate (is this necessary?)
    Q = expm(dt*mat)

    # Advance F
    F = np.dot(Q,np.dot(F,Q.T))
    Fs.append(F)



