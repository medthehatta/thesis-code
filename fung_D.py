# coding: utf-8
"""
fung_D.py

Doesn't actually compute the tangent stiffness *per se*, but computes that part
of the tangent stiffness that could be nonpositive-definite.
"""
import numpy as np
import sympy as sp

# Construct the quadratic form
Q = np.empty((6,6),dtype=object)
for i in range(6):
    for j in range(i,6):
        Q[i,j]=sp.symbols('q_{i}{j}'.format(i=i,j=j))
        Q[j,i]=Q[i,j]
# Flat list of independent entries of Q
q = sum([Q[i,i:].tolist() for i in range(len(Q))],[])

# Construct the Lagrangian strain (E) as a *vector* (e)
f = [sp.symbols('f_{i}'.format(i=i)) for i in range(9)]
F = np.reshape(f,(3,3))
E = 0.5*(np.dot(F.T,F) - np.eye(3))
e = np.array(sum([E[i,i:].tolist() for i in range(len(E))],[]))

# Expand the quadratic form's action on e
Qee = np.dot(e,np.dot(Q,e))

# Construct the tangent stiffness as a symbolic expression
D_symbolic = np.empty((9,9),dtype=object)
D_numeric = np.empty((9,9),dtype=object)
for i in range(9):
    for j in range(i,9):
        D_symbolic[i,j] = sp.diff(Qee,f[i])*sp.diff(Qee,f[j])+sp.diff(Qee,f[i],f[j])
        D_symbolic[j,i] = D_symbolic[i,j]
        # Transform each symbolic expression into a python function
        D_numeric[i,j] = sp.lambdify(q+f,D_symbolic[i,j],np)
        D_numeric[j,i] = D_numeric[i,j]

# Take the arguments to be passed to each function "component" of the tangent stiffness
def D_function(*args): return np.array([[D[i,j](*args) for D in DD] for DD in D_numeric])

