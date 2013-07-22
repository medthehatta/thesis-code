#!/usr/bin/env python3
# coding: utf8
#

import argparse
import os.path
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



def uniaxial_loading(s):
    return np.diagflat([s, 1/np.sqrt(s), 1/np.sqrt(s)])

def equibiaxial_loading(s):
    return np.diagflat([s,s,1/(s*s)])

def pure_shear_loading(stretch1,stretch2):
    fix_J = lambda mat: lin.direct_sum(mat,np.diagflat([1/np.linalg.det(mat)]))
    return fix_J(np.array([[1,s/2.],[s/2.,1]]))



def load(loading, start, end=None):
    if end is None: 
        return loading(start)
    else:
        return np.array([loading(s) for s in np.linspace(start,end,200)])


def compute_vars(F, H, model, *params):
    def pressure(F, H, *params):
        So = model.iso_material_model(F,H,*params)
        sigma = np.dot(F,np.dot(So,F.T))
        return sigma[-1,-1]

    def get_S(F, p, H, *params):
        C = np.dot(F.T,F)
        Ci = np.linalg.inv(C)
        return -p*Ci + model.iso_material_model(F,H,*params)

    def get_CC(F, p, H, *params):
        Co = model.iso_material_elasticity(F,H,*params)
        # pt = p: assuming dp/dJ=0
        Cv = model.vol_material_elasticity(F,p,p,H,*params)
        return Cv + Co

    def get_AA(CC, F, S):
        return np.einsum('mjnl,kn,im',CC,F,F) +\
               lin.kronecker(np.eye(3),S)


    # Deformation
    I = np.eye(3)
    C = [np.dot(f.T,f) for f in F]
    B = [np.dot(f,f.T) for f in F]
    E = [0.5*(c - I) for c in C]
    Is = [[np.trace(c), 0.5*(np.trace(c) - np.trace(np.dot(c,c)))] for c in C]

    # Energy
    W = [model.strain_energy_density(f,H,*params) for f in F]

    # Stress
    press = [pressure(f,H,*params) for f in F]
    S = [get_S(f,p,H,*params) for (f,p) in zip(F,press)]
    P = [np.dot(f,s) for (f,s) in zip(F,S)]
    sigma = [np.dot(p,f.T) for (f,p) in zip(F,P)]

    # Elasticity
    CC = [get_CC(f,p,H,*params) for (f,p) in zip(F,press)]
    AA = [get_AA(cc,f,s) for (cc,f,s) in zip(CC,F,S)]

    # Return this behemoth in a dictionary
    return {'F':F, 'C':C, 'B':B, 'E':E, 'Is':Is, 'W':W, 'press':press, 'S':S,
            'P':P, 'sigma':sigma, 'CC':CC, 'AA':AA}
    
    


def plot_loading_curves(params,loading,title="",start=1,end=1.5,prefix="/tmp"):
    Ws = np.array([strain_energy(F,*params) 
                   for F in load(loading,start,end)])

    Ps = np.array([constitutive_model(F,*params) 
                  for F in load(loading,start,end)])

    return plot_W_P(Ws,Ps,title,prefix)



def plot_W_P(Ws,Ps,title="",prefix="/tmp"):
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
    subpath = "G_{}.png".format(rand)
    plt.savefig(os.path.join(prefix,subpath))
    print(os.path.join(prefix,subpath))

    return (Ws,Ps,subpath)

def plot_loadingspec(params, loadingspec):
    loading = loadingspec.get('loading')
    title = loadingspec.get('title')
    start = loadingspec.get('start') or 1.0
    end = loadingspec.get('end') or 1.4
    prefix = loadingspec.get('prefix') or "/tmp"
    return plot_loading_curves(params,loading,title,start,end,prefix)

LOADINGS = { 'uniaxial':uniaxial_loading,
             'equibiaxial':equibiaxial_loading,
             'shear':pure_shear_loading }


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('loading', choices=list(LOADINGS.keys()),
                        help="loading mode")

    parser.add_argument('parameters', type=float, nargs='+',
                        help="model parameters")

    parser.add_argument('-s', '--start', type=float, default=1.0,
                        help="initial loading parameter")

    parser.add_argument('-e', '--end', type=float, default=1.5,
                        help="final loading parameter")

    parser.add_argument('-t', '--title', default="Deformation path",
                        help="title to use on plots")

    parser.add_argument('-p', '--prefix', default="/tmp",
                        help="directory plots are output to")

    args = parser.parse_args()

    plot_loading_curves(params=args.parameters,
                        loading=LOADINGS.get(args.loading),
                        title=args.title,
                        start=args.start,
                        stop=args.end)

