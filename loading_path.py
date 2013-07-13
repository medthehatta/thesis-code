import matplotlib.pyplot as plt
import sys
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
dt = 0.01

# Let's try equibiaxial loading, maybe?
Ws = []
EVs = []
for l in np.arange(1,1.3,0.01):
    F = np.diagflat([l,l,1/(l*l)])
    W = strain_energy(F,*params)
    Ws.append(W)

plt.cla()

major_title = (" ".join(sys.argv[1:])).title()
minor_title = "dt: {}   p: {}".format(dt,params)
plt.suptitle(major_title+"\n"+minor_title)

plt.subplot(211)
plt.plot(Ws)

plt.subplot(212)
plt.plot(EVs)

rand = np.random.randint(99999)
path = "stuff/test_plots/G_{}.png".format(rand)
plt.savefig("/home/med/astro/public_html/"+path)
print("http://astro.temple.edu/~tud48344/"+path)



