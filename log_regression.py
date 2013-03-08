"""
log_regression.py

Logistic regression binary classifier.
"""
import numpy as np
import scipy.optimize as so

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
    return so.fmin(cost,np.ones(X.shape[-1]),dcost)

