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

def initialize(symfile_path="fung_Dsym.pkl"):
    """
    Returns the tangent stiffness function, computed symbolically at runtime.
    """
    # Construct the quadratic form as a flat list of independent entries
    q = np.array([sp.symbols('q_{}{}'.format(i,j)) for 
                  (i,j) in lin.utri_indices(6)])

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
        Dsym = pickle.load(open(symfile_path,'rb'))

    # Otherwise, just recompute it
    except Exception:
        # Construct the tangent stiffness as a symbolic expression

        # Calculate first derivatives
        dQ = np.empty(9,dtype=object)
        for i in range(9):
            print("Symbolic dQ_{}".format(i))
            dQ[i] = sp.diff(Qee,f[i])

        # Calculate second derivatives
        Dsym = np.empty((9,9),dtype=object)
        symfile = open(symfile_path,'wb')
        for (i,j) in lin.utri_indices(9):
            print("Symbolic ddQ_{0}{1}".format(i,j))
            dQi = dQ[i]
            dQj = dQ[j]
            dQij = sp.diff(dQi,f[j])
            Dsym[i,j] = dQi*dQj + dQij

            # Optimize the derivative by substituting for J, and for
            # products of f components
            print("  Simplifying...")
            print("  J  ",end="")
            sys.stdout.flush()
            Dsym[i,j] = Dsym[i,j].subs(J,sp.symbols('J'))
            for (k,l) in lin.utri_indices(9):
                print("f{}f{}".format(k,l),end="  ")
                sys.stdout.flush()
                pair_symbol = sp.symbols('ff_{0}{1}'.format(k,l))
                Dsym[i,j] = Dsym[i,j].subs(f[k]*f[l],pair_symbol)
            # Since D will be symmetric, assign the symmetric components
            print("\n  Symmetrizing...")
            Dsym[j,i] = Dsym[i,j]
            # This computation is pretty costly, so let's save it
            # frequently
            pickle.dump(Dsym, symfile)
        symfile.close()

    return Dsym

def make_numeric(Dsym):
    # Transform each symbolic expression into a python function
    # We'll need the products of f components
    # Construct the quadratic form
    Q = np.empty((6,6),dtype=object)
    for (i,j) in lin.utri_indices(6):
        Q[i,j]=sp.symbols('q_{i}{j}'.format(i=i,j=j))
        Q[j,i]=Q[i,j]

    # Flat list of independent entries of Q
    q = lin.utri_flat(Q)

    # Components of F
    f = np.array([sp.symbols('f_{i}'.format(i=i)) for i in range(9)])

    # Products of F components
    ff = np.array([sp.symbols('ff_{}{}'.format(i,j)) for (i,j) in lin.utri_indices(9)])

    # Put everything into the numeric array
    Dnum = np.empty((9,9),dtype=object)
    for (i,j) in lin.utri_indices(9):
        print("Numeric D_{0}{1}".format(i,j))

        # This will take the quadratic form, f, products of f, and J as
        # arguments
        arguments = [ob.tolist() for ob in [q,f,ff]] + \
                    [[sp.symbols('J')]]
        Dnum[i,j] = sp.lambdify(sum(arguments,[]),Dsym[i,j])
        Dnum[j,i] = Dnum[i,j]

    return Dnum

# Pass a deformation F and a set of parameters (c,b1...b9)
def D(F,c,bs,Dnum):
    J = np.linalg.det(F)
    f = F.ravel()
    ff = lin.utri_flat(np.outer(f,f))
    C = fung.make_quad_form(*bs)
    q = lin.utri_flat(el.voigt(C)[:6,:6])
    arglist = sum([ob.tolist() for ob in [q,f,ff]+[[J]]],[])
    return c*np.array([[dnum(*arglist) for dnum in DD] for DD in Dnum])

