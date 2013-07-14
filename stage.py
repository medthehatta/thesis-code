import numpy as np
import lintools as lin
import mooney_rivlin as mr
import elastic as el

import parse_kaveh_compression as pkc
import mooney_rivlin as mr
import scipy.optimize as so

import log_regression as lr

import argparse


def det1_3d(mat2d):
    J0 = np.linalg.det(mat2d)
    return lin.direct_sum(mat2d,np.diagflat([1/J0]))



def random_F(scale=1.0):
    F0 = np.eye(2) + scale*np.random.random((2,2))
    return det1_3d(F0)



def general_pressure_PK1(F,constitutive_model,*params,vanishing=(-1,-1)):
    """
    Get the pressure from the constitutive model.
    constitutive_model(F,pressure,*params)
    I hope this works.
    """
    return el.pressure_PK1(F,constitutive_model,*params,P=0,component=vanishing)



def cost_kaveh(params, lam=1e2, lam2=1., debug=False):
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

    # Regularization
    regularize = sum([p*p for p in params])

    if debug==True:
        return (total_error, penalty, lam*penalty, regularize, lam2*regularize)
    else:
        return total_error + lam*penalty + lam2*regularize



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



def analyze_params_mr(params):
    def tstiff(F,*p):
        pressure = general_pressure_PK1(F,mr.constitutive_model,*p)
        return mr.material_tangent_stiffness(F,pressure,*p)


    identity = np.eye(3)
    identity_result = test_drucker(params,tstiff,[identity])
    id_true = bool(len(identity_result[0]))

    # biaxial
    regional = 0.01 + (1.6-0.01)*np.random.random((800,2))
    regional_mats = [np.diagflat([r1,r2,1/(r1*r2)]) for (r1,r2) in regional]
    region_result = test_drucker(params,tstiff,regional_mats)
    r_num_true = len(region_result[0])
    r_num_false = len(region_result[1])
    r_pct_true = 100*r_num_true/(r_num_true + r_num_false)
    r_pct_false = 100*r_num_false/(r_num_true + r_num_false)

    (trues,falses,_) = region_result
    values = np.array([1]*len(trues) + [0]*len(falses))
    samples_m = np.concatenate([f for f in [trues,falses] if len(f)>0])
    samples = np.diagonal(samples_m,axis1=1,axis2=2)[:,:2] - [1,1]
    poly_samples = lr.monomialize_vector(samples, lr.dim2_deg4[:,None,:])
    cal = lr.calibrate_logistic(poly_samples, values, lam=0.1)

    print("Stable at identity: " + str(id_true).upper())
    print("Eigensystem at identity:")
    (eigvals,eigvecs) = np.linalg.eigh(el.voigt(tstiff(np.eye(3),*params)))
    for (eigva,eigve) in zip(eigvals,eigvecs.T):
        print("{}\n  {}".format(eigva,eigve))

    r1_t = "Stable at {} and unstable at {} of {} points sampled from region ({}%)"
    r1_text = r1_t.format(r_num_true,r_num_false,r_num_true+r_num_false,r_pct_true)
    print(r1_text)

    print("4th Order Classifier:")
    for (monom,coeff) in zip(lr.dim2_deg4,cal['x']):
        print("{}  {}".format(monom,coeff))

    return (cal['x'],np.array(trues),np.array(falses))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters',type=float,nargs='+',
                        help='mooney-rivlin model parameters to test')
    args = parser.parse_args()
    analyze_params_mr(args.parameters)

