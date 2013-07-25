"""
stable_check.py
"""
import numpy as np
import lintools as lin




def test_condition_at_pts(condition,points,params=None):
    """
    Performs a test for ``condition`` at the supplied ``points``.

    If the condition also depends on a fixed parameter, pass that through
    ``params``.

    Return the points that met the condition, those that didn't, and a list
    labeling each point with true or false.
    """
    if params is None:
        res = [condition(pt) for pt in points]
    else:
        res = [condition(pt,*params) for pt in points]

    labeled = list(zip(points,res))
    trues = [p for (p,r) in labeled if r==True]
    falses = [p for (p,r) in labeled if r!=True]
    return (trues,falses,labeled)


def points_in_box(low,hi,num=1000):
    """
    Samples ``num`` points from a box in R^n whose lower corner is at ``low``
    and whose opposite corner is at ``hi``.
    """
    return low + (hi - low)*np.random.random([num]+list(low.shape))


def det_J1(mat):
    """
    Takes a matrix and returns the unit determinant matrix obtained by adding
    the reciprocal determinant to the diagonal (as a direct sum).
    """
    return lin.direct_sum(mat,np.diagflat([1/np.linalg.det(mat)]))





