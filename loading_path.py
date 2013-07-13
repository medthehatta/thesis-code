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

params = [-1.01,1.49]
dt = 0.001

F = np.eye(3)
Ws = []
EVs = []
counts = 0
while np.abs(F[0,0])<3 and 1/np.abs(F[0,0])>0.3 and counts<1000:
    counts+=1
    W = strain_energy(F,*params)
    print(W)
    Ws.append(W)

    D = el.voigt(tangent_stiffness(F,*params))
    (eigvals,eigvecs) = np.linalg.eigh(D)
    positive_directions = [i for (i,v) in zip(count(),eigvals) if v>0]
    direction = eigvecs[positive_directions[-1]]
    F += dt*vec_to_mat(direction)
    EVs.append(eigvals[positive_directions[-1]])
    
rand = np.random.randint(99999)
print(rand)

plt.cla()
plt.plot(Ws)
plt.savefig("/home/med/astro/public_html/stuff/test_plots/W_{}.png".format(rand))

plt.cla()
plt.plot(EVs)
plt.savefig("/home/med/astro/public_html/stuff/test_plots/EV_{}.png".format(rand))


