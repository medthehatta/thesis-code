# coding: utf-8
"""
fung_D.py

Doesn't actually compute the tangent stiffness *per se*, but computes that part
of the tangent stiffness that could be nonpositive-definite.
"""
import numpy as np
import sympy as sp
import elastic as el
import lintools as lin
import fung
import pickle
import sys

def initialize():
    """
    Returns the tangent stiffness function, computed symbolically at runtime.
    """
    # Construct the quadratic form
    Q = np.empty((6,6),dtype=object)
    for i in range(6):
        for j in range(i,6):
            Q[i,j]=sp.symbols('q_{i}{j}'.format(i=i,j=j))
            Q[j,i]=Q[i,j]
    # Flat list of independent entries of Q
    q = lin.utri_flat(Q)

    # Construct the Lagrangian strain (E) as a *vector* (e)
    f = np.array([sp.symbols('f_{i}'.format(i=i)) for i in range(9)])
    F = np.reshape(f,(3,3))
    J = sp.Matrix(F.tolist()).det()
    # TODO: Is this supposed to be J**(-4/3) or J**(-2/3)?
    E = 0.5*(J**sp.Rational(-2,3)*np.dot(F.T,F) - np.eye(3))
    e = lin.utri_flat(E)

    # Expand the quadratic form's action on e
    Qee = np.dot(e,np.dot(Q,e))

    # Attempt to load this from a pickle file
    try:
        D_symbolic = pickle.load(open("fung_D_symbolic.pkl",'rb'))

    # Otherwise, just recompute it
    except Exception:
        # Construct the tangent stiffness as a symbolic expression

        # Calculate first derivatives
        dQ = np.empty(9,dtype=object)
        for i in range(9):
            print("Symbolic dQ_{}".format(i))
            dQ[i] = sp.diff(Qee,f[i])

        # Calculate second derivatives
        D_symbolic = np.empty((9,9),dtype=object)
        symfile = open("fung_D_symbolic.pkl",'wb')
        for i in range(9):
            for j in range(i,9):
                print("Symbolic ddQ_{0}{1}".format(i,j))
                dQi = dQ[i]
                dQj = dQ[j]
                dQij = sp.diff(dQi,f[j])
                D_symbolic[i,j] = dQi*dQj + dQij

                # Optimize the derivative by substituting for J, and for
                # products of f components
                print("  Simplifying...")
                print("  J  ",end="")
                sys.stdout.flush()
                D_symbolic[i,j].subs(J,sp.symbols('J'))
                for k in range(9):
                    for l in range(k,9):
                        print("f{}f{}".format(k,l),end="  ")
                        sys.stdout.flush()
                        pair_symbol = sp.symbols('ff_{0}{1}'.format(k,l))
                        D_symbolic[i,j] = D_symbolic[i,j].subs(f[k]*f[l],pair_symbol)
                # Since D will be symmetric, assign the symmetric components
                print("\n  Symmetrizing...")
                D_symbolic[j,i] = D_symbolic[i,j]
                # This computation is pretty costly, so let's save it frequently
                pickle.dump(D_symbolic, symfile)
        symfile.close()

    # Transform each symbolic expression into a python function
    # We'll need the products of f components
    ff = lin.utri_flat(np.outer(f,f))
    D_numeric = np.empty((9,9),dtype=object)
    for i in range(9):
        for j in range(i,9):
            print("Numeric D_{0}{1}".format(i,j))
            arguments = [ob.tolist() for ob in [q,f,ff]] + [sp.symbols('J')]
            D_numeric[i,j] = sp.lambdify(arguments,D_symbolic[i,j])
            D_numeric[j,i] = D_numeric[i,j]
    return (D_symbolic,D_numeric)

# Pass a deformation F and a set of parameters (c,b1...b9)
def D(F,c,bs,D_numeric):
    f = F.ravel()
    C = fung.make_quad_form(*bs)
    CC = lin.utri_flat(el.voigt(C)[:6,:6])
    arglist = CC.tolist() + f.tolist()
    return np.array([[D(*arglist) for D in DD] for DD in D_numeric])

