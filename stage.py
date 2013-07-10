import numpy as np
import lintools as lin
import mooney_rivlin as mr
import elastic as el

import parse_kaveh_compression as pkc
import mooney_rivlin as mr
import scipy.optimize as so

def uniaxial_pressure(F,*params2):
    """
    Uniaxial pressure
    Assume F is diagonal.
    """
    (c10,c01) = params2

    l = F[0,0]
    ll = l*l

    # Construct invariants from stretch
    I1 = ll + 2/l
    I2 = 1/ll + 2*l

    # Assemble the awful pressure expression
    p1 = 1/l 
    p2 = I1/l - 1/ll

    return 2*(c10*p1 + c01*p2)

def cost_kaveh(params, lam=1e2, lam2=1., debug=False):
    # Collate the data
    data = zip(pkc.deformations, pkc.PK1)

    # Least square error
    def uniaxial_MR(f,*params):
        return mr.constitutive_model(f,uniaxial_pressure(f,*params),*params)
    errors = np.array([P[0,0] - uniaxial_MR(f,*params)[0,0] for 
                       (f,P) in data])
    total_error = np.dot(errors,errors)

    # Penalty error
    if lam>0:
        tests = [test_mr_drucker(c,uniaxial_pressure(c,*params),*params) 
                 for c in pkc.right_cauchy_green]
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
    penalty_parameters = [0,1,10,100,1000]
    setups = [[ini+[n] for n in penalty_parameters] for ini in initials]
    fits = automatic_fits(sum(setups,[]),cost_kaveh,min_method,reg)
    return [(k,fits[k]['x'],fits[k]['fun']) for k in fits.keys()]

def test_mr_drucker(C,pressure,*params):
    def tstiff(C,*p):
      return mr.material_tangent_stiffness(C,uniaxial_pressure(C,*p),*p)

    trues = test_drucker(params,tstiff,[C])[0]
    return len(trues)>0

def test_drucker(params,tangent_stiffness,points):
    res = [lin.is_positive_definite(el.voigt(tangent_stiffness(pt,*params)))
           for pt in points]
    labeled = list(zip(points,res))
    trues = [p for (p,r) in labeled if r==True]
    falses = [p for (p,r) in labeled if r==False]
    return (trues,falses,labeled)
    
def analyze_params_uniaxial_mr(params):
    def tstiff(C,*p):
      return mr.material_tangent_stiffness(C,uniaxial_pressure(C,*p),*p)

    identity = np.eye(3)
    regional = 0.01 + 2.0*np.random.random(100)
    regional_mats = [np.diagflat([r,1/np.sqrt(r),1/np.sqrt(r)]) for r in regional]

    identity_result = test_drucker(params,tstiff,[identity])
    id_true = bool(len(identity_result[0]))

    region_result = test_drucker(params,tstiff,regional_mats)
    r_num_true = len(region_result[0])
    r_num_false = len(region_result[1])
    r_pct_true = 100*r_num_true/(r_num_true + r_num_false)
    r_pct_false = 100*r_num_false/(r_num_true + r_num_false)

    data_result = test_drucker(params,tstiff,pkc.right_cauchy_green)
    d_num_true = len(data_result[0])
    d_num_false = len(data_result[1])
    d_pct_true = 100*d_num_true/(d_num_true + d_num_false)
    d_pct_false = 100*d_num_false/(d_num_true + d_num_false)

    id_text = "Stable at identity: " + str(id_true).upper()

    r1_t = "Stable at {} and unstable at {} of {} points sampled from region ({}%)"
    r1_text = r1_t.format(r_num_true,r_num_false,r_num_true+r_num_false,r_pct_true)

    d1_t = "Stable at {} and unstable at {} of {} data points ({}%)"
    d1_text = d1_t.format(d_num_true,d_num_false,d_num_true+d_num_false,d_pct_true)

    print("\n".join([str(params),id_text,r1_text,d1_text]))

    if d_pct_true == 100:
        print("== STABLE OVER DATA ==")
    else:
        print("(Unstable over data)")

    return sorted([(np.trace(c), np.trace(np.dot(c,c)), r) for (c,r) in region_result[2]],key=lambda x:x[-1])

