#!/usr/bin/env python3
# coding: utf8

# Python modules
import pandas as pd
import numpy as np
import scipy.io
from pandas.io.parsers import read_csv

# My modules
import lintools as lin


# Global variables
PREFIX = "/home/med/re/data/kaveh-compression/"

# Read in data
df = read_csv(PREFIX+"kaveh-brain-compression.tab", sep='\t',
              header=None, names=['l0','e','P (kPa)'], comment='#').dropna()

# l0 = prestretch
# e = -strain from prestretch
# P = PK1 stress
(l0,e,P) = df.as_matrix().T

# Calculate the total stretch 
l = [(1-ee)*ll0 for (ll0,ee) in zip(l0,e)]

# Calculate the deformation gradient assuming the other directions are
# unconstrained and that the sample is incompressible
deformations = [np.diagflat([ll]+[1./np.sqrt(ll)]*2) for ll in l]

# Cauchy-Green
left_cauchy_green = [np.dot(f,f.T) for f in deformations]
right_cauchy_green = [np.dot(f.T,f) for f in deformations]

# Calculate the full PK1 stress assuming the other directions are stress-free
PK1 = [np.diagflat([-p]+[0,0]) for p in P]

# Calculate the cauchy stresses from the PK1 stresses
cauchy = [np.dot(p,f.T) for (p,f) in zip(PK1,deformations)]

# Vector Cauchy
v3Cauchy = [np.diag(cauch) for cauch in cauchy]


