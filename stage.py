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

def test_mr_drucker(C,*params):
    tstiff = el.voigt(mr.material_tangent_stiffness(C,*params))
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
import mooney_rivlin as mr

def biaxial_MR(b, *params):
    """
    Incompressibility gives pressure from boundary condition that radial
    boundaries are stress-free
    """
    cauchy0 = mr.constitutive_model(b,*params)
    return cauchy0 - cauchy0[-1,-1]*np.eye(3)

def cost(params, lam=1e2):
    # Collate the data
    data = zip(png.left_cauchy_green_p, png.v3Cauchy)

    # Least square error
    errors = np.array([sigma - np.diag(biaxial_MR(b,*params)) for 
                       (b,sigma) in data])
    total_error = np.tensordot(errors,errors) / np.dot(errors[0],errors[0])

    # Penalty error
    tests = [test_mr_drucker(c,*params) for c in png.right_cauchy_green_p]
    penalty = tests.count(False)/len(tests)

    return total_error + lam*penalty

# so.minimize(cost, [200,-50], args=(100,), callback=print, method='Powell')
