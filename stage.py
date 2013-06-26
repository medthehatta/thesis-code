import numpy as np
import lintools as lin

def random_F(scale=1.0):
    F0 = np.eye(2) + scale*np.random.random((2,2))
    return det1_3d(F0)

def symmetric_2_1(a11,a22,a12):
    return np.array([[a11,a12],[a12,a22]])

def det1_3d(mat2d):
    J0 = np.linalg.det(mat2d)
    return lin.direct_sum(mat2d,np.diagflat([1/J0]))

