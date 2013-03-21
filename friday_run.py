import numpy as np
import matplotlib.pyplot as plt
import log_regression as reg
def test_logistic(pts,normal,lam=0.0):
   firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
   seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
   trues = np.tile(True,(pts,))
   falses = np.tile(False,(pts,))
   plt.scatter(*firsts[:,1:].T,color='green')
   plt.scatter(*seconds[:,1:].T,color='red')
   result = reg.calibrate_logistic(np.vstack([firsts,seconds]),
                                       np.hstack([trues,falses]),
                                       lam=lam)
   if len(result)==3:
     x = np.arange(0,2,0.1)
     y = np.arange(0,1,0.1)
     (X,Y) = np.meshgrid(x,y)
     XY = np.array([np.ones(X.shape),X,Y])
     TH = np.tile(result,[1]+list(XY.shape[1:]))
     # TODO: tile result so TH has XY.shape
     DOTS = np.einsum('i...,i...',TH,XY)
     Z = reg.sigmoid(DOTS)
     plt.contour(X,Y,Z)
   nrm = np.linalg.norm(result[1:])
   print("{0} + {2}{1}".format(result[0],result[1:]/nrm,nrm))
   return result

#test_logistic(500,[1,0])
#test_logistic(5000,[1,0])
#test_logistic(500000,[1,0])
#test_logistic(50000,[1,0])
#test_logistic(50,[1,0])
#test_logistic(5,[1,0])
