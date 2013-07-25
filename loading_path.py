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


def fix_J(mat):
    return lin.direct_sum(mat,np.diagflat([1/np.linalg.det(mat)]))

def uniaxial_loading(s):
    return np.diagflat([s, 1/np.sqrt(s), 1/np.sqrt(s)])

def uniaxial_loading2(s):
    return np.diagflat([1/np.sqrt(s), s, 1/np.sqrt(s)])

def uniaxial_loading3(s):
    return fix_J(np.diagflat([0.7*(s-(1-1/0.7)), 1.1*(s - (1-1/1.1))]))

def equibiaxial_loading(s):
    return np.diagflat([s,s,1/(s*s)])

def equibiaxial_loading2(s):
    return fix_J(np.diagflat([s*np.cos(np.pi/3.),s*np.sin(np.pi/3.)]))

def pure_shear_loading(s):
    return fix_J(np.array([[1,s/2.],[s/2.,1]]))



def load(loading, start, end=None):
    if end is None: 
        return loading(start)
    else:
        points = np.linspace(start,end,200)
        return [points, np.array([loading(s) for s in points])]


def compute_vars(loads, H, model, *params):
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


    # Get the loading parameter and deformation
    (par, F) = loads

    # Deformation
    I = np.eye(3)
    C = [np.dot(f.T,f) for f in F]
    B = [np.dot(f,f.T) for f in F]
    E = [0.5*(c - I) for c in C]

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
    return {'F':F, 'C':C, 'B':B, 'E':E, 'W':W, 'press':press, 'S':S, 'P':P,
            'sigma':sigma, 'CC':CC, 'AA':AA, 'par':par}
    
    


def plot_loading_curves(params,loading,title="",start=1,end=1.5,prefix="/tmp"):
    Ws = np.array([strain_energy(F,*params) 
                   for F in load(loading,start,end)])

    Ps = np.array([constitutive_model(F,*params) 
                  for F in load(loading,start,end)])

    return plot_W_P(Ws,Ps,title,prefix)



def plot_curve(VARS,title="",prefix="/tmp",filename=None):
    plt.clf()

    fig = plt.figure(figsize=(8.5,11))
    major_title = (title.title())
    fig.suptitle(major_title, fontsize=14, fontweight='bold')


    # Energy density (added without any precalc necessary)

    # PK1 invariants
    I1p = [np.trace(P) for P in VARS['P']]
    I2p = [0.5*(np.trace(P)**2 - np.trace(np.dot(P,P))) for P in VARS['P']]

    # PK2 invariants
    I1s = [np.trace(S) for S in VARS['S']]
    I2s = [0.5*(np.trace(S)**2 - np.trace(np.dot(S,S))) for S in VARS['S']]

    # Principal stretches
    (stre1,stre2,stre3) = np.transpose([np.linalg.eigvalsh(c) for c in VARS['C']])

    # Pressure (added without any precalc necessary)


    # Add the data to the plot
    to_plot = []
    to_plot.append(["Energy Density", [(VARS['W'], None)]])
    to_plot.append(["Principal Stretches", [(np.sqrt(stre1),r'$\lambda_1$'),
                                            (np.sqrt(stre2),r'$\lambda_2$'),
                                            (np.sqrt(stre3),r'$\lambda_3$')]])

    max_PK = max(I1p+I2p+I1s+I2s)
    min_PK = min(I1p+I2p+I1s+I2s)
    to_plot.append(["PK1 Invariants", [(I1p, r'$I_1$'), (I2p, r'$I_2$')], (min_PK,max_PK)])

    to_plot.append(["PK2 Invariants", [(I1s, r'$I_1$'), (I2s, r'$I_2$')], (min_PK,max_PK)])
    #to_plot.append(["Pressure", [(VARS['press'],None)]])


    # Make a list of axes
    my_axes = []
    for v in range(len(to_plot)):
        if v>0:
            my_axes.append(fig.add_subplot(len(to_plot),1,v+1,sharex=my_axes[0]))
        else:
            my_axes.append(fig.add_subplot(len(to_plot),1,v+1))

    # Actually plot each item
    for (ax, plot_setup) in zip(my_axes,to_plot):
        
        # Get the title, plots, and maybe ylimits
        (ax_title,plots) = plot_setup[:2]
        if len(plot_setup)==3:
            ax.set_ylim(*plot_setup[-1])
        ax.set_ylabel(ax_title)

        # Loop through the plots in each item
        for (plot_dat,plot_lab) in plots:
            if plot_lab is not None:
                ax.plot(VARS['par'],plot_dat,label=plot_lab)
            else: 
                ax.plot(VARS['par'],plot_dat)

        # If the x-axis is *de*creasing, flip
        if VARS['par'][0] > VARS['par'][-1]:
           ax.invert_xaxis() 

        # Set some common axis parameters
        ax.tick_params(labelsize=10)
        ax.grid(True)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1.1,0.5))

    # Generate an ID or use the name and save the plot
    if filename is None:
        rand = np.random.randint(99999)
        subpath = "G_{}.png".format(rand)
    else:
        subpath = filename
    plt.savefig(os.path.join(prefix,subpath))
    print(os.path.join(prefix,subpath))

    return subpath






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

