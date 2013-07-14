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



def load(loading, start, end):
    return np.array([loading(s) for s in np.linspace(start,end,200)])



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

