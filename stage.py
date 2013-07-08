import numpy as np
import lintools as lin
import mooney_rivlin as mr
import elastic as el

def random_F(scale=1.0):
    F0 = np.eye(2) + scale*np.random.random((2,2))
    return det1_3d(F0)

def symmetric_2x2(a11,a22,a12):
    return np.array([[a11,a12],[a12,a22]])

def det1_3d(mat2d):
    J0 = np.linalg.det(mat2d)
    return lin.direct_sum(mat2d,np.diagflat([1/J0]))

def test_mr_drucker(C,pressure,*params):
    tstiff = el.voigt(mr.material_tangent_stiffness(C,pressure,*params))
    return lin.is_positive_definite(tstiff)

def points_in_box(lower=np.zeros(2),upper=np.ones(2),num=10):
    result_shape = [num]+list(lower.shape)
    signs = np.random.choice([-1,1],size=result_shape)
    return lower + (upper-lower)*signs*np.random.random(result_shape)

def stable_region(lower,upper,*p,num=10):
    pts = points_in_box(lower,upper,num)
    Bs = [det1_3d(np.eye(2) + symmetric_2x2(*pt)) for pt in pts]
    tests = [test_mr_drucker(b,*p) for b in Bs]
    return (tests.count(True)/len(tests))

# For constrained fitting

import parse_new_golriz as png
import parse_kaveh_compression as pkc
import mooney_rivlin as mr
import scipy.optimize as so

def biaxial_MR(b, *params):
    """
    Incompressibility gives pressure from boundary condition that radial
    boundaries are stress-free
    """
    cauchy0 = mr.constitutive_model(b,pressure=0,*params)
    return cauchy0 - cauchy0[-1,-1]*np.eye(3)

def uniaxial_MR(b, *params):
    return mr.constitutive_model(b,uniaxial_pressure(b,*params),*params)

def uniaxial_pressure(C,*params2):
    """
    Uniaxial pressure
    Assume c is in principal coordinates
    """
    (c10,c01,c11) = params2
    ll = C[0,0]
    l = np.sqrt(ll)

    # Construct invariants from stretch
    I1 = ll + 2/l
    I2 = 1/ll + 2*l

    # Assemble the awful pressure expression
    p1 = 1/l 
    p2 = I1/l - 1/ll
    p3 = (I2/l + I1*I1/l - I1/ll)/3 + 1/ll - 1/l - I1/l

    return 2*c10*p1 + 2*c01*p2 + 6*c11*p3

def cost_golriz(params, lam=1e2):
    # Collate the data
    data = zip(pkc.left_cauchy_green_p, pkc.v3Cauchy)

    # Least square error
    errors = np.array([sigma - np.diag(biaxial_MR(b,*params)) for 
                       (b,sigma) in data])
    total_error = np.tensordot(errors,errors) / np.dot(errors[0],errors[0])

    # Penalty error
    tests = [test_mr_drucker(c,*params) for c in png.right_cauchy_green_p]
    penalty = tests.count(False)/len(tests)

    # Regularization
    regularize = sum([p*p for p in params])

    return total_error + lam*penalty + lam2*regularize

def cost_kaveh(params, lam=1e2, lam2=1.):
    # Collate the data
    data = zip(pkc.left_cauchy_green, pkc.v3Cauchy)

    # Least square error
    errors = np.array([sigma - np.diag(uniaxial_MR(b,*params)) for 
                       (b,sigma) in data])
    total_error = np.tensordot(errors,errors) / np.dot(errors[0],errors[0])

    # Penalty error
    if lam>0:
        tests = [test_mr_drucker(c,uniaxial_pressure(c,*params),*params) j
                 for c in pkc.right_cauchy_green]
        penalty = tests.count(False)/len(tests)
    else:
        penalty = 0

    # Regularization
    regularize = sum([p*p for p in params])

    return total_error + lam*penalty + lam2*regularize

def automatic_fits(setups,cost):
    results = {}
    for (initial1,initial2,initial3,lam) in setups:
        results[(initial1,initial2,initial3,lam)] = \
            so.minimize(cost, [initial1,initial2,initial3], args=(lam,), \
                        callback=print, method='Powell')
    return results

def sweep_auto_fits(initials,cost):
    penalty_parameters = [0,1,10,100,1000]
    setups = [[(a,b,c,n) for n in penalty_parameters] for (a,b,c) in initials]
    fits = automatic_fits(sum(setups,[]),cost_kaveh)
    return [(k,fits[k]['x'],fits[k]['fun']) for k in fits.keys()]

some_initials = [\
                 (-1.01,1.49,0.19), # kaveh
                 (200,-50,0),
                 (200,100,0),
                 (10,10,0),
                 (1,1,0),
                 (-100,100,0)]


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

    id_text = "Stable at identity: " + str(id_true).upper()

    r1_t = "Stable at {} and unstable at {} of {} points sampled from region ({}%)"
    r1_text = r1_t.format(r_num_true,r_num_false,r_num_true+r_num_false,r_pct_true)
    
    if r_pct_false==0:
        r_text = "Stable over all samples from region: TRUE"
    else:
        r_text = "Stable over all samples from region: FALSE"

    print("\n".join([str(params),id_text,r1_text,r_text]))

    return sorted([(np.trace(c), np.trace(np.dot(c,c)), r) for (c,r) in region_result[2]],key=lambda x:x[-1])

