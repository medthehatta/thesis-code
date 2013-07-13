import matplotlib.pyplot as plt
import sys
import numpy as np
import elastic as el
import lintools as lin
from itertools import count

# Use mooney-rivlin
import mooney_rivlin as mr
constitutive_model = lambda F, *p: mr.constitutive_model(F, el.pressure_PK1(F, mr.constitutive_model, *p), *p)
tangent_stiffness = lambda F, *p: mr.material_tangent_stiffness(F, el.pressure_PK1(F, mr.constitutive_model, *p), *p)
strain_energy = lambda F, *p: mr.strain_energy_density(F, *p)


def uniaxial_loading(stretch1,stretch2):
    return np.array([np.diagflat([l,1/np.sqrt(l),1/np.sqrt(l)]) 
                     for l in np.linspace(stretch1,stretch2,200)])

def equibiaxial_loading(stretch1,stretch2):
    return np.array([np.diagflat([l,l,1/(l*l)]) 
                     for l in np.linspace(stretch1,stretch2,200)])

def pure_shear_loading(stretch1,stretch2):
    fix_J = lambda mat: lin.direct_sum(mat,np.diagflat([1/np.linalg.det(mat)]))
    return np.array([fix_J(np.array([[1,l/2.],[l/2.,1]]))
                     for l in np.linspace(stretch1,stretch2,200)])

def plot_loading_curves(params,loading,title="",start=1,stop=1.5):
    Ws = np.array([strain_energy(F,*params) for F in loading(start,stop)])
    Ps = np.array([constitutive_model(F,*params) for F in loading(start,stop)])

    plt.clf()

    (f, (ax1, ax2)) = plt.subplots(2, sharex=True)

    major_title = (title.title())
    f.suptitle(major_title)

    ax1.set_ylabel("Energy Density")
    ax1.plot(Ws)
    ax1.tick_params(labelsize=10)
    ax1.grid(True)

    ax2.set_ylabel("PK1 Stress Invariants")
    I1s = [np.trace(P) for P in Ps]
    I2s = [0.5*(np.trace(P)**2 - np.trace(np.dot(P,P))) for P in Ps]
    Js = [np.linalg.det(P) for P in Ps]
    ax2.plot(I1s,label=r'$I_1$')
    ax2.plot(I2s,label=r'$I_2$')
    ax2.tick_params(labelsize=10)
    ax2.grid(True)

    rand = np.random.randint(99999)
    path = "stuff/test_plots/G_{}.png".format(rand)
    plt.savefig("/home/med/astro/public_html/"+path)
    print("http://astro.temple.edu/~tud48344/"+path)

    return (Ws,Ps,plt)


