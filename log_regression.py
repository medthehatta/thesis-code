"""
log_regression.py

Logistic regression binary classifier.
"""
import numpy as np
import scipy.optimize as so
from itertools import permutations

def sigmoid(X):
    """Compute the sigmoid function"""
    den = 1.0 + np.exp(-X)
    d = 1.0 / den
    return d


def compute_cost(theta, X, y, lam=0.0):
    """
    Compute cost for logistic regression
    """
    # Number of training samples
    m = y.shape[0]

    # Hypothesis and complement
    h = sigmoid(np.dot(X,theta))
    hc = 1.0 - h

    # Actually we just care about their logs
    lh = np.log(h)
    lhc = np.log(hc)

    # Compute complementary y-values
    yc = 1.0 - y

    # Cost function: J = -1/m yi log(hi) - 1/m yci log(hci) + l/2m(theta^2)
    # N.B.: you don't regularize the theta_0 (bias) term
    J = -(1.0/m)*(np.dot(y,lh) + np.dot(yc,lhc)) + \
         (lam/(2*m))*np.dot(theta[1:],theta[1:])
    return J


def compute_grad(theta, X, y, lam=0.0):
    """
    Gradient of cost function
    """
    # Number of training samples
    m = y.shape[0]

    # Hypothesis 
    h = sigmoid(np.dot(X,theta))

    # Regularization term is kinda weird here
    # (Weirdness results from not regularizing the theta_0 (bias) term
    reg = lam*np.hstack([0,theta[1:]])

    # Gradient 1/m (h-y) x + regularization
    dJ = (1.0/m)*(np.dot((h - y),X) + reg)
    return dJ


def calibrate_logistic(X, y, lam=0.0):
    """
    Train a linear logistic classifier.  Return the parameter vector.
    """
    cost = lambda t: compute_cost(t,X,y,lam)
    dcost = lambda t: compute_grad(t,X,y,lam)
    return so.minimize(cost,np.ones(X.shape[-1]),jac=dcost,method="CG")


# Pre-made powers for monomializing 2d vectors
# The [0,0] guarantees a bias term for calibration of the logistic classifier
dim2_deg2 = np.array([[0,0],[1,0],[0,1],[2,0],[0,2],[1,1]])
dim2_deg3 = np.array([[0,0],[1,0],[0,1],[2,0],[0,2],[1,1],[3,0],[0,3],[2,1],[1,2]])
dim2_deg4 = np.array([[0,0],[1,0],[0,1],[2,0],[0,2],[1,1],[3,0],[0,3],[2,1],[1,2],[4,0],[0,4],[3,1],[1,3],[2,2]])

def monomialize_vector(vec,powers):
    """
    Takes a vector and raises its elements to various powers, combining them
    into monomials and returning a vector of them.
    I.E., [x1, x2] -> [x1, x2, x1^2 x2, x1 x2^2, ...]
    """
    transposed = np.product(vec**powers,axis=-1)
    return np.rollaxis(transposed,0,len(transposed.shape))

def evaluate_poly(poly_coeffs,powers,vector):
    return np.dot(monomialize_vector(vector, powers), poly_coeffs)

