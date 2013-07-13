import matplotlib.pyplot as plt
import numpy as np
import elastic as el
import lintools as lin
from itertools import count

def vec_to_mat(vec):
    (n1,n2,n3,s1,s2,s3) = vec
    return np.array([[n1,s3,s2],[s3,n2,s1],[s2,s1,n3]])

def general_pressure_PK1(F,constitutive_model,*params,vanishing=(1,1)):
    """
    Get the pressure from the constitutive model.
    constitutive_model(F,pressure,*params)
    I hope this works.
    """
    P_nop = constitutive_model(F,0,*params)
    pI = np.dot(P_nop,F.T)
    return pI[vanishing]


# Use mooney-rivlin
import mooney_rivlin as mr
constitutive_model = lambda F, *p: mr.constitutive_model(F, general_pressure_PK1(F, mr.constitutive_model, *p), *p)
tangent_stiffness = lambda F, *p: mr.material_tangent_stiffness(F, general_pressure_PK1(F, mr.constitutive_model, *p), *p)
strain_energy = lambda F, *p: mr.strain_energy_density(F, *p)

params = [2.01,-1.49]
dt = 0.01

F = np.eye(3)
Ws = []
EVs = []
for j in range(700):
    W = strain_energy(F,*params)
    Ws.append(W)

    D = el.voigt(tangent_stiffness(F,*params))
    (eigvals,eigvecs) = np.linalg.eigh(D)
    positive_directions = [i for (i,v) in zip(count(),eigvals) if v>0]
    EVs.append(eigvals[positive_directions[0]])
    direction = eigvecs[positive_directions[0]]

    F += dt*vec_to_mat(direction)



plt.cla()
plt.subplot(211)
plt.plot(Ws)

plt.subplot(212)
plt.plot(EVs)


plt.show()


