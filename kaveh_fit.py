#!/usr/bin/env python3
# coding: utf8
#

import numpy as np
import lintools as lin
import mooney_rivlin as mr
import elastic as el

import parse_kaveh_compression as pkc
import mooney_rivlin as mr
import scipy.optimize as so

import log_regression as lr

import argparse


def general_pressure_PK1(F,constitutive_model,*params,vanishing=(-1,-1)):
    """
    Get the pressure from the constitutive model.
    constitutive_model(F,pressure,*params)
    I hope this works.
    """
    return el.pressure_PK1(F,constitutive_model,*params,P=0,component=vanishing)



def cost_kaveh(params, lam=1e2, lam2=1., lamp=500., debug=False):
    # Collate the data
    data = zip(pkc.deformations, pkc.PK1)

    # Least square error
    def MR(f,*params):
        pressure = general_pressure_PK1(f,mr.constitutive_model,*params)
        return mr.constitutive_model(f,pressure,*params)
    errors = np.array([P[0,0] - MR(f,*params)[0,0] for (f,P) in data])
    total_error = np.dot(errors,errors)

    # Penalty error
    if lam>0:
        tests = [test_mr_drucker(f,*params) for f in pkc.deformations]
        penalty = tests.count(False)/len(tests)
    else:
        penalty = 0

    # Penalty error 2: "physicality"
    if sum(params)<0:
        penalty2 = lamp
    else:
        penalty2 = 0

    # Regularization
    regularize = sum([p*p for p in params])

    if debug==True:
        return (total_error, penalty, penalty2, regularize)
    else:
        return total_error + lam*penalty + lam2*regularize + penalty2



def automatic_fits(setups,cost,min_method='Powell',reg=1.0):
    results = {}
    for setup in setups:
        results[tuple(setup)] = \
            so.minimize(cost, setup[:-1], args=(setup[-1],reg), \
                        callback=print, method=min_method)
    return results



def sweep_auto_fits(initials,cost,min_method='Powell',reg=1.0):
    penalty_parameters = [0,1,10,100,1e6]
    setups = [[ini+[n] for n in penalty_parameters] for ini in initials]
    fits = automatic_fits(sum(setups,[]),cost_kaveh,min_method,reg)
    return [(k,fits[k]['x'],fits[k]['fun']) for k in fits.keys()]



def test_drucker(params,tangent_stiffness,points):
    res = [lin.is_positive_definite(el.voigt(tangent_stiffness(pt,*params)))
           for pt in points]
    labeled = list(zip(points,res))
    trues = [p for (p,r) in labeled if r==True]
    falses = [p for (p,r) in labeled if r==False]
    return (trues,falses,labeled)
    


def test_mr_drucker(F,*params):
    def tstiff(F,*p):
        pressure = general_pressure_PK1(F,mr.constitutive_model,*p)
        return mr.material_tangent_stiffness(F,pressure,*p)

    trues = test_drucker(params,tstiff,[F])[0]
    return len(trues)>0



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters',type=float,nargs='+',
                        help='mooney-rivlin model parameters to test')
    args = parser.parse_args()
    analyze_params_mr(args.parameters)

