import matplotlib.pyplot as plt
import sys
import numpy as np
import elastic as el
import lintools as lin
from itertools import count

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


def uniaxial_loading(stretch1,stretch2):
    return np.array([np.diagflat([l,1/np.sqrt(l),1/np.sqrt(l)]) 
                     for l in np.linspace(stretch1,stretch2,200)])

def equibiaxial_loading(stretch1,stretch2):
    return np.array([np.diagflat([l,l,1/(l*l)]) 
                     for l in np.linspace(stretch1,stretch2,200)])

# I hope that stress-free unloaded direction is the correct B.C.
(e1,e2,e3) = np.eye(3)
def pure_shear(stretch1,stretch2):
    return np.array([l*lin.symmetric(np.outer(e1,e2))
                     for l in np.linspace(stretch1,stretch2,200)])


def plot_loading_curves(params,loading,title="",start=1,stop=1.5):
    Ws = [strain_energy(F,*params) for F in loading(start,stop)]
    Ps = [constitutive_model(F,*params) for F in loading(start,stop)]

    plt.cla()

    major_title = (title.title())
    minor_title = "dt: {}   p: {}".format(dt,params)
    plt.suptitle(major_title+"\n"+minor_title)

    plt.subplot(211)
    plt.title("Strain energy")
    plt.plot(Ws)

    plt.subplot(212)
    plt.title("First component of PK1 stress")
    plt.plot(Ps)

    rand = np.random.randint(99999)
    path = "stuff/test_plots/G_{}.png".format(rand)
    plt.savefig("/home/med/astro/public_html/"+path)
    print("http://astro.temple.edu/~tud48344/"+path)

    return (Ws,Ps)


