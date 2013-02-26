# coding: utf-8
from imp import reload
import numpy as np
import data_process
import xlrd
import matplotlib.pyplot as plt
book = xlrd.open_workbook("/home/med/re/data/golriz-biaxial/h4t1.xls")
nolo_pqrs = [1,-1]*np.reshape(book.sheet_by_name("NoLoPQRS").row_values(0),(4,2))
pqrs = np.array([[book.sheet_by_name('DynPQRS').col_values(i),book.sheet_by_name('DynPQRS').col_values(j)] for (i,j) in [[0,1],[2,3],[4,5],[6,7]]]).transpose(2,0,1)
def plot_deformation(verts):
    ax =plt.gcf().add_subplot(111)
    for v in verts:
        poly=plt.Polygon(v,fill=False)
        ax.add_patch(poly)
    plt.axis('scaled')
    plt.show()
rtf = data_process.relative_to_first

