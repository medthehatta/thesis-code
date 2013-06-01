import numpy as np
import fung_D
import fung
import lintools as lin
import pickle

Dsym = pickle.load(open("Dsym_withJ.pkl",'rb'))
Dnum = fung_D.make_numeric(Dsym)

stables = open("stable_parameters",'w')
num_pts = 4000
for F in np.eye(3)+1.2*np.random.random((num_pts,3,3)):
    D = fung_D.D(F, 1.0, fung.STABLE_IN_E[1], Dnum)
    print(F.ravel())
    print(D)
    if lin.is_positive_definite(D):
        print("##### STABLE!!!!")
        msg = "{0}\n{1}".format(p,F.ravel())
        print(msg)
        stables.write("\n\n"+msg)
    else:
        print("unstable")
    print("\n\n")
stables.close()

