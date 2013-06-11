#!/usr/bin/env python3
# coding: utf8

# Python modules
import pandas as pd
import numpy as np
import scipy.io
from pandas.io.parsers import ExcelFile

# My modules
import lintools as lin


# Global variables
PREFIX = "/home/med/re/data/golriz-biaxial/"


# Helper functions
def F_between(u1,u2,x1,x2):
    """
    Computes the deformation gradient at the center of a line between two
    nodes.  Since the data is only 2d, the 3d deformation gradient is
    constructed by assuming incompressibility and perfectly planar loading.

    Parameters
    ----------
    u1 : displacement at one node
    u2 : displacement at another node
    x1 : material position of the first node
    x2 : material position of the second node

    Returns
    -------
    3x3 deformation gradient between the two nodes
    """

    # F = I + Grad(u)
    # Grad(u) is approximately: (delta u)_i * (1/delta x)_j
    delta_u = u1 - u2
    delta_x_recip = 1.0/(x1 - x2)
    grad_u = np.outer(delta_u,delta_x_recip)
    
    # 2x2 portion of F
    F0 = np.eye(2) + grad_u

    # Compute the determinant of F so we know how much the radial direction
    # should contract
    J = np.linalg.det(F0)

    # F is then F0(+)(1/J)
    # (where (+) means direct sum)
    return lin.direct_sum(F0,np.diagflat([1/J]))





# Load data from data files
DCOORD = ExcelFile(PREFIX+"D_Coord.xlsx")
BASE = scipy.io.loadmat(PREFIX+"ZState.mat")
FORCE = scipy.io.loadmat(PREFIX+"Force_h4t3.mat")

# Dimensions of the sample (stress-free?  prestretch?)
(Llc,Lll,thk) = BASE['ZDim'].tolist()[0]

# Positions of markers in (stress-free?  prestretch?) configuration
# material_coords[marker, C/L]
material_coords = BASE['ZCoord'].reshape((11,2))

# Circumferential and Longitudinal force at each timestep
# force_data[timestep, C/L]
force_data = np.array([FORCE['ForceC'][:,0],FORCE['ForceL'][:,0]]).T

# Position of each marker at each timestep
# deformation_coords[timestep, marker, C/L]
deformation_coords = DCOORD.parse("h4t3",header=1,parse_cols="Y:AT")\
                           .as_matrix()
# Split up the flat point coordinates into a list of 2d points
new_dc_shape = (len(deformation_coords), 11, 2)
deformation_coords = deformation_coords.reshape(new_dc_shape)

# Get the scale taking deformation coords from pixels to mm
deformation_scale = BASE['scale']

# Rescale the deformation coordinates
deformation_coords = deformation_scale*deformation_coords

##################################################
# Compute the Lagrangian strain at each timestep # 
##################################################

# Displacements of markers at each timestep
# displacements[timestep, marker, C/L]
displacements = deformation_coords - material_coords

# Compute the deformation gradient for each timestep by averaging the
# deformation gradients on edges which overlap roughly in the center
antipodal_nodes = [(1,11),(2,10),(3,8),(4,7),(5,9)]
# Convert to 0-indexed instead of 1-indexed
antipodal_nodes = [(i-1,j-1) for (i,j) in antipodal_nodes]

# Find the deformation gradients at each edge for each timestep
deformations = np.array([[F_between(u[i],u[j],
                                    material_coords[i],material_coords[j]) 
                 for (i,j) in antipodal_nodes] 
                 for u in displacements])

# Average the deformation gradients for all the edges at each timestep to
# get the average deformation gradient at that timestep
avg_deformations = np.average(deformations,axis=1)

# Compute the Lagrangian strain for each deformation
right_cauchy_green = np.einsum('...ab,...ac',avg_deformations
                                            ,avg_deformations)
lagrangian_strain = 0.5*(right_cauchy_green - np.eye(3))


###########################################
# Compute the PK1 stress at each timestep # 
###########################################

# First, compute the areas that the forces are being applied to to get the
# principal stresses
# The units are (mm)^2, but we want to convert to SI, so we divide by 10^6
# EXCEPT we don't want Pa, we want kPa, so we only divide by 10^3!
areas = (thk*np.array([Llc,Lll,1]))/1e3

# We assume that the stress is perfectly biaxial
PK1 = np.array([np.diagflat(f.tolist()+[0]) for f in force_data])/areas

# Let's also make a PK2 version so we can compare it with Fung more easily
PK2 = np.array([np.dot(F,P) for (F,P) in zip(avg_deformations,PK1)])




