# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
def many_randpoints(N,d=3,s=100): return 10*np.random.random((s,N,d))
def mc_test(num,d=3,s=100):
    R = many_randpoints(num,d=d,s=s)
    L = np.logical_and(R>2,R<7)
    counts = np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
    estimate = counts/num * 10**d
    return np.average(estimate), np.std(estimate)
def stddev_estimate(xmax=200.,ymax=500.,xstep=5.,ystep=10.):
  plt.title('Monte Carlo Integration Error Estimate (StdDev)')
  plt.xlabel('Number of samples per stddev')
  plt.ylabel('Number of monte carlo sample points')
  X=np.arange(5,xmax+1,xstep)
  Y=np.arange(5,ymax+1,ystep)
  plt.pcolor(X, Y, np.array([[mc_test(i,3,s)[1] for s in X] for i in Y]),cmap=plt.cm.gray_r)
