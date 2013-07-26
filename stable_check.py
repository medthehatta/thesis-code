"""
stable_check.py
"""
import numpy as np
import lintools as lin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

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


def plot_3d_condition_test(trues,falses):
    """
    Given trues and falses in R^3, plots an orthographic view of the point
    cloud.
    """
    plt.clf()
    gs = gridspec.GridSpec(2,2, width_ratios=[1,1])

    ax_y = plt.subplot(gs[1,0])
    ax_x = plt.subplot(gs[1,1], sharey=ax_y)
    ax_z = plt.subplot(gs[0,0], sharex=ax_y)
    ax_iso = plt.subplot(gs[0,1], projection='3d')
    
    ax_x.axis('equal')
    ax_x.set_aspect(1.0)
    ax_y.axis('equal')
    ax_y.set_aspect(1.0)
    ax_z.axis('equal')
    ax_z.set_aspect(1.0)
    
    ax_x.set_title("Right (YZ)")
    ax_y.set_title("Front (XZ)")
    ax_z.set_title("Top (XY)")
    ax_iso.set_title("3d Scatter")

    ax_iso.set_xlabel("X")
    ax_iso.set_ylabel("Y")
    ax_iso.set_zlabel("Z")

    plt.setp(ax_x.get_yticklabels(), visible=False)
    plt.setp(ax_z.get_xticklabels(), visible=False)
    plt.setp(ax_iso.get_xticklabels(), visible=False)
    plt.setp(ax_iso.get_yticklabels(), visible=False)
    plt.setp(ax_iso.get_zticklabels(), visible=False)
    
    if len(falses)>0:
        (fX,fY,fZ) = np.transpose(falses)
        ax_x.scatter(fY,fZ,color='red')
        ax_y.scatter(fX,fZ,color='red')
        ax_z.scatter(fX,fY,color='red')
        ax_iso.scatter(fX,fY,fZ,color='red',alpha=0.1)
    
    if len(trues)>0:
        (tX,tY,tZ) = np.transpose(trues)
        ax_x.scatter(tY,tZ,color='green')
        ax_y.scatter(tX,tZ,color='green')
        ax_z.scatter(tX,tY,color='green') 
        ax_iso.scatter(tX,tY,tZ,color='green',alpha=0.8)
 
    return plt.gcf()



def plot_2d_condition_test(trues,falses):
    """
    Given trues and falses in R^2, plots a view of the point cloud.
    """
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    
    if len(falses)>0:
        (fX,fY) = np.transpose(falses)
        ax.scatter(fX,fY,color='red')
    
    if len(trues)>0:
        (tX,tY) = np.transpose(trues)
        ax.scatter(tX,tY,color='green') 
 
    return plt.gcf()



