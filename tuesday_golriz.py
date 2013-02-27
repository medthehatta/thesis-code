# coding: utf-8
from imp import reload
import numpy as np
import data_process
import xlrd
import matplotlib.pyplot as plt

book      = xlrd.open_workbook("/home/med/re/data/golriz-biaxial/h4t1.xls")

nolo_pqrs = [1,-1]*np.reshape(book.sheet_by_name("NoLoPQRS").row_values(0),(4,2))

pqrs_book = book.sheet_by_name('DynPQRS')
pqrs      = np.reshape([pqrs_book.row_values(i) for i in range(pqrs_book.nrows)],(pqrs_book.nrows,4,2))

(Lx,Ly,Lz) = book.sheet_by_name("Dimension").row_values(0)

forces = np.transpose([book.sheet_by_name("Force").col_values(0),book.sheet_by_name("Force").col_values(1)])

def plot_deformation(verts):
    ax =plt.gcf().add_subplot(111)
    for v in verts:
        poly=plt.Polygon(v,fill=False)
        ax.add_patch(poly)
    plt.axis('scaled')
    plt.show()

rtf = data_process.relative_to_first


# remove redundant vertex: now we have simplices
nolo_pqs = np.delete(nolo_pqrs,2,0)
pqs = np.delete(pqrs,2,1)

# translate all simplices to origin
nolo_pqs0 = rtf(nolo_pqs)
pqs0 = rtf(pqs,1)

# transpose each entry to get gross deformation gradient
F0 = nolo_pqs0.T
F1 = pqs0.swapaxes(1,2)

# find deformations relative to F0
#  F1 = F.F0 -> F = F1.F0i
F0i = np.linalg.inv(F0)
F   = np.einsum('...ij,...jk',F1,F0i)

# find piola kirchhoff stress from force and areas
pressures = [Lx*Lz,Ly*Lz]*forces
PK1       = np.array([np.diagflat(f) for f in forces])

