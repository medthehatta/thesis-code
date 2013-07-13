import matplotlib.pyplot as plt
import sys
import numpy as np
import elastic as el
import lintools as lin
from itertools import count

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

def pure_shear_loading(stretch1,stretch2):
    return np.array([np.array([[0,l/2.,0],[l/2.,0,0],[0,0,4./(l*l)]])
                     for l in np.linspace(stretch1,stretch2,200)])

def plot_loading_curves(params,loading,title="",start=1,stop=1.5):
    Ws = np.array([strain_energy(F,*params) for F in loading(start,stop)])
    Ps = np.array([constitutive_model(F,*params) for F in loading(start,stop)])

    plt.clf()

    major_title = (title.title())
    plt.suptitle(major_title)

    plt.subplot(211)
    plt.title("Strain energy")
    plt.plot(Ws)

    plt.subplot(212)
    plt.title("First component of PK1 stress")
    plt.plot(Ps[:,0,0])

    rand = np.random.randint(99999)
    path = "stuff/test_plots/G_{}.png".format(rand)
    plt.savefig("/home/med/astro/public_html/"+path)
    print("http://astro.temple.edu/~tud48344/"+path)

    return (Ws,Ps)


