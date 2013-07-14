import numpy as np
from scipy.linalg import expm
import lintools as lin
import elastic as el

import mooney_rivlin as mr

def tstiff(F, *p):
    pressure = el.pressure_PK1(F, mr.constitutive_model, *p)
    return mr.material_tangent_stiffness(F, pressure, *p)

def const(F, *p):
    pressure = el.pressure_PK1(F, mr.constitutive_model, *p)
    return mr.constitutive_model(F, pressure, *p)

def sedf(F, *p):
    return mr.strain_energy_density(F, *p)

def voigt_vec_to_mat(d):
    return np.array([[d[0],d[5],d[4]],
                     [d[5],d[1],d[3]],
                     [d[4],d[3],d[2]]]) 


dt = 0.003
params = [3.,-1.]

F = np.eye(3)
Fs = [F]
selecteds = []
for j in range(200):
    # Compute D
    D = el.voigt(tstiff(F,*params))

    # Find a positive eigendirection
    (eigvals, eigvecs) = np.linalg.eigh(D)
    eigsystem = zip(eigvals,eigvecs)
    candidates = [vec for (val,vec) in eigsystem if val>0]
    if len(candidates)==0: break

    # Try to pick the eigenvector with the closest eigenvalue to the last
    # picked
    if selecteds != []:
        diffs = [np.abs(c-selected[-1]) for c in candidates]
        selected = candidates[np.argmin(diffs)]
        selecteds.append(selected)
    else:
        selected = candidates[0]

    # Convert to matrix
    mat = voigt_vec_to_mat(selected)

    # Try adding instead of adjoint action
    F = F + dt*mat
    Fs.append(F)

Ws = [sedf(F,*params) for F in Fs]
Ps = [const(F,*params) for F in Fs]


