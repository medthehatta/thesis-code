import numpy as np
import lintools as lin

def random_F(scale=1.0):
    F0 = np.eye(2) + scale*np.random.random((2,2))
    J0 = np.linalg.det(F0)
    F = lin.direct_sum(F0,np.diagflat([1/J0]))
    return F


