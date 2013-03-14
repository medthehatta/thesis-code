# coding: utf-8

def many_randpoints(N): return 10*np.random.random((100,N,3))
many_randpoints(600)
np.logical_and(Out[2]>2,Out[2]<7)
R = many_randpoints(600)
np.logical_and(R>2,R<7)
_.shape
get_ipython().magic('pinfo np.logical_and')
get_ipython().magic('pinfo np.logical_and')
np.roll(R,-1,axis=2)
L np.logical_and(R>2,R<7)
L = np.logical_and(R>2,R<7)
L
np.roll(L,-1,axis=2)
np.logical_and(np.logical_and(L,np.roll(L,-1,axis=2)),np.roll(L,-2,axis=2))
np.logical_and(np.logical_and(L,np.roll(L,-1,axis=2)),np.roll(L,-2,axis=2))[:,:,0]
L
np.ones(5)[[True,False,False,True,True]]
np.ones(5)[np.array([True,False,False,True,True])]
L
np.ones(L.shape)[L]
np.ones(L.shape)
np.ones(L.shape)[L]
_.shape
L
L[L]=1
L[not L]=0
L
L[L==True]=1
L
True + False
True + True
get_ipython().magic('pinfo np.sum')
np.sum(L,axis=2)
_[_==3]
Out[33]==3
get_ipython().magic('pinfo np.count_nonzero')
np.sum(Out[33]==3,axis=-1)
Out[33]
R = many_randpoints(600)
L = np.logical_and(R>2,R<7)
np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
L = np.logical_and(R>2,R<7)
np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
np.average(_), np.std(_)
R = many_randpoints(1200)
L = np.logical_and(R>2,R<7)
np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
np.average(_), np.std(_)
def mc_test(num):
    R = many_randpoints(num)
    L = np.logical_and(R>2,R<7)
    counts = np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
    estimate = counts/num * 1000.
    return np.average(estimate), np.std(estimate)
mc_test(20)
mc_test(40)
mc_test(80)
mc_test(160)
mc_test(320)
plt.plot([mc_test(i)[0] for i in range(1,1000)])
plt.plot([mc_test(i)[0] for i in range(1,1000)])
plt.cla()
plt.plot(range(1,1000),[mc_test(i)[0] for i in range(1,1000)])
plt.plot(range(1,1000), 1/np.sqrt(range(1,1000)))
plt.cla()
plt.plot(range(1,1000),[mc_test(i)[1] for i in range(1,1000)])
plt.plot(range(1,1000), 1/np.sqrt(range(1,1000)))
plt.cla()
#plt.plot(range(1,1000),[mc_test(i)[1] for i in range(1,1000)])
plt.plot(range(1,1000), 1/np.sqrt(range(1,1000)))
def many_randpoints(N,d=3): return 10*np.random.random((100,N,d))
def mc_test(num,d=3):
    R = many_randpoints(num,d)
    L = np.logical_and(R>2,R<7)
    counts = np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
    estimate = counts/num * 1000.
    return np.average(estimate), np.std(estimate)
plt.plot(range(1,1000),[mc_test(i,2)[1] for i in range(1,1000)])
plt.plot(range(1,100),[mc_test(i,2)[1] for i in range(1,100)])
plt.plot(range(1,300),[mc_test(i,2)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,3)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,1)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,2)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,3)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,4)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,5)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,6)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,1)[1] for i in range(1,300)],color='red')
plt.plot(range(1,300),[mc_test(i,2)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,3)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,4)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,5)[1] for i in range(1,300)])
plt.plot(range(1,300),[mc_test(i,6)[1] for i in range(1,300)])
for d in range(1,6+1):    
    plt.plot(range(1,300),[mc_test(i,d)[1] for i in range(1,300)],label=str(d))
plt.legend()
for d in range(1,6+1):    
    plt.plot(range(1,300),[mc_test(i,d)[0] for i in range(1,300)],label=str(d))
plt.legend()
def mc_test(num,d=3):
    R = many_randpoints(num,d)
    L = np.logical_and(R>2,R<7)
    counts = np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
    estimate = counts/num * 10**d
    return np.average(estimate), np.std(estimate)
for d in range(1,6+1):    
    plt.plot(range(1,300),[mc_test(i,d)[0] for i in range(1,300)],label=str(d))
plt.legend()
for d in range(1,5+1):    
    plt.plot(range(1,300),[mc_test(i,d)[0] for i in range(1,300)],label=str(d))
plt.legend()
for d in range(1,5+1):    
    plt.plot(range(1,300),[mc_test(i,d)[1] for i in range(1,300)],label=str(d))
plt.legend()
def many_randpoints(N,d=3,s=100): return 10*np.random.random((s,N,d))
def mc_test(num,d=3,s=100):
    R = many_randpoints(num,d=d,s=s)
    L = np.logical_and(R>2,R<7)
    counts = np.sum(np.sum(L,axis=-1)==L.shape[-1],axis=-1)
    estimate = counts/num * 10**d
    return np.average(estimate), np.std(estimate)
for s in 10*np.range(1,15):    
    plt.plot(range(1,300),[mc_test(i,3,s)[1] for i in range(1,300)],label=str(s))
plt.legend()
for s in 10*np.arange(1,15+1):    
    plt.plot(range(1,300),[mc_test(i,3,s)[1] for i in range(1,300)],label=str(s))
plt.legend()
[np.std([mc_test(i,3,s) for s in 10*np.arange(1,15+1)]) for i in range(1,20)]
[[mc_test(i,3,s) for s in 10*np.arange(1,15+1)]) for i in range(1,20)]
[[mc_test(i,3,s) for s in 10*np.arange(1,15+1)] for i in range(1,20)]
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in) range(1,20)]
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in range(1,20)])
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in range(1,20)])
np.set_printoptions(linewidth=150,precision=3,suppress=True)
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in range(1,20)])
_.shape
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in 10*np.arange(1,20)])
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in np.arange(1,20,10)])
np.array([[mc_test(i,3,s)[0] for s in 10*np.arange(1,15+1)] for i in np.arange(1,200,10)])
get_ipython().magic('pinfo np.std')
np.average(_,axis=0)
np.average(__,axis=1)
np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,15+1)] for i in np.arange(1,200,10)])
_.shape
np.set_printoptions(linewidth=160,precision=3,suppress=True)
np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,15+1)] for i in np.arange(1,200,10)])
np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,15+1)] for i in np.arange(1,500,10)])
plt.contour(_)
plt.pcolor(Out[97])
plt.pcolor(Out[97][1:])
plt.pcolor(Out[97][2:])
plt.pcolor(Out[97][2:],cmap=plt.cm.gray_r)
plt.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(1,500,10)]),cmap=plt.cm.gray_r)
plt.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(3,500,10)]),cmap=plt.cm.gray_r)
fig = plt.figure(dpi=120)
fig.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(3,500,10)]),cmap=plt.cm.gray_r)
fig = plt.figure(dpi=120)
ax = fig.axes()
ax.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(3,500,10)]),cmap=plt.cm.gray_r)
plt.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylavel('Number of monte carlo sample points')
plt.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
plt.pcolor(np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
plt.pcolor(10*np.arange(1,20+1), np.arange(5,500,10), np.array([[mc_test(i,3,s)[1] for s in 10*np.arange(1,20+1)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
plt.pcolor(np.arange(2,200+1), np.arange(5,500,10), np.array([[mc_test(i,3,s)[1] for s in np.arange(2,200+1,10)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
plt.pcolor(np.arange(2,200+1,10), np.arange(5,500,10), np.array([[mc_test(i,3,s)[1] for s in np.arange(2,200+1,10)] for i in np.arange(5,500,10)]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
X=np.arange(2,200+1,5)
Y=np.arange(5,500,10)
plt.pcolor(X, Y, np.array([[mc_test(i,3,s)[1] for s in X] for i in Y]),cmap=plt.cm.gray_r)
plt.title('Monte Carlo Integration Error Estimate (StdDev)')
plt.xlabel('Number of samples per stddev')
plt.ylabel('Number of monte carlo sample points')
X=np.arange(5,200+1,5)
Y=np.arange(5,500,10)
plt.pcolor(X, Y, np.array([[mc_test(i,3,s)[1] for s in X] for i in Y]),cmap=plt.cm.gray_r)