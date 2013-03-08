# coding: utf-8

get_ipython().magic('run log_regression.py')
samples = np.hstack([np.ones((20,1)), np.random.random((20,3))])
theta = np.array([1,.3,.2,.5])
predict(theta,samples)
theta = np.array([0.1,.01,.2,.5])
predict(theta,samples)
values = np.random.choice([0,1],samples.shape[0])
import scipy.optimize as so
so.fmin_bfgs(lambda T: compute_cost(T,samples,values), np.random.random(theta.shape[0]), lambda T: compute_grad(T,samples,values))
theta_c = _
predict(theta_c,samples)
values
sigmoid(np.dot(theta_c,samples))
sigmoid(np.dot(samples, theta_c))
_[_>0.5]
import stable_check
import stable_check
import stable_check
import fung
import strain
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),1.0,0.333)
reload(stable_check)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),1.0,0.333)
reload(stable_check)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),1.0,0.333)
reload(stable_check)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),1.0,0.333)
_[0].shape[0]
__[1].shape[0]
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.1*np.ones(2),0.2*np.ones(2),1.0,0.333)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.1*np.ones(2),0.2*np.ones(2),1.0,0.333,ptsdensity=20)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.1*np.ones(2),0.2*np.ones(2),1.0,0.333,ptsdensity=200)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.1*np.ones(2),0.2*np.ones(2),1.0,0.333,ptsdensity=2000)
len(_[0]),len(_[1])
import holzapfel
stable_check.positive_definite_samples(holzapfel.model_D, strain.biaxial_extension_vec,0.1*np.ones(2),0.2*np.ones(2),0.0,0.2,0.333,ptsdensity=2000)
(len(_[0]),len(_[1]))
def positive_D(F,*params):
    dim = F.shape[-1]
    return params[0]*np.einsum('...ij,...kl->...ikjl',np.eye(dim),np.eye(dim))
positive_D(np.eye(3),6)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
positive_D(np.eye(3),5)
import lintools as lin
lin.np_voigt(_)
lin.is_positive_definite(_)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
F
F = np.random.random((50,3,3))
lin.np_voigt(F[0])
F = np.random.random((50,3,3,3,3))
lin.np_voigt(F[0])
F[0]
np.reshape(F, (F.shape[0],9,9))
_.shape
__[0] == Out[52]
reload(lin)
F
lin.np_voigt_vec(F)
__[0] == Out[52]
__[0] == Out[52]
reload(datafit)
get_ipython().magic('whos')
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
get_ipython().magic('pdb')
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
def positive_D(F,*params):
    dim = F.shape[-1]
    I = np.tile(np.eye(dim), (F.shape[0],1,1))
    return params[0]*np.einsum('...ij,...kl->...ikjl',I,I)
def positive_D(F,*params):
    dim = F.shape[-1]
    I = np.tile(np.eye(dim), (F.shape[0],1,1))
    return params[0]*np.einsum('...ij,...kl->...ikjl',I,I)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),0.0,0.2,0.333,ptsdensity=1.0)
(len(_[0]),len(_[1]))
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)
(len(_[0]),len(_[1]))
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)
plt.scatter(Out[75].T[0],Out[75].T[1],'ro')
get_ipython().magic('pdb')
plt.scatter(Out[75][0].T[0],Out[75][0].T[1],'ro')
Out[75][0].T[0]
_.shape
Out[75][0].T[1]
_.shape
plt.scatter(Out[75][0].T[0],Out[75][0].T[1])
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0]
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0]
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0]
plt.scatter(*_.T)
reload(stable_check)
stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0]
plt.scatter(*_.T)
plt.scatter(stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0].T)
plt.scatter(*stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0].T)
plt.scatter(*stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,-10*np.ones(2),10*np.ones(2),2.0,ptsdensity=1.0)[0].T)
plt.scatter(*stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=2.0)[0].T)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=20.0)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=200.0)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=2000.0)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=20000.0)
stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
CHK = stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
plt.scatter(*CHK[0].T)
plt.scatter(*CHK[1].T)
CHK = stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
plt.scatter(*CHK[0].T)
plt.scatter(*CHK[1].T)
CHK = stable_check.positive_definite_samples(fung.model_isotropic_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
plt.cla()
plt.scatter(*CHK[0].T,color='green')
plt.scatter(*CHK[1].T,color='red')
CHK = stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
plt.cla()
plt.scatter(*CHK[0].T,color='green')
plt.scatter(*CHK[1].T,color='red')
reload(stable_check)
CHK = stable_check.positive_definite_samples(positive_D, strain.biaxial_extension_vec,0.01*np.ones(2),0.1*np.ones(2),10.0,0.2,ptsdensity=8000.0)
plt.cla()
plt.scatter(*CHK[0].T,color='green')
plt.scatter(*CHK[1].T,color='red')
import interface
interface.run_from_args(*interface.argument_parse("-F dat/F.txt -P dat/P.txt -M dat/M.txt".split()))
reload(reg)
import log_regression as log
del log
import log_regression as reg
get_ipython().system('g st')
get_ipython().system('git status')
import log_regression as reg
reload(reg)
reg.calibrate_logistic(np.vstack([np.random.random((20,2)),1+np.random.random((20,2))]),np.tile(True,(20,)))
reload(reg)
reg.calibrate_logistic(np.vstack([np.random.random((20,2)),1+np.random.random((20,2))]),np.tile(True,(20,)))
reg.calibrate_logistic(np.vstack([np.random.random((20,2)),1+np.random.random((20,2))]),np.vstack([np.tile(True,(20,)),np.tile(False,(20,))]))
reg.calibrate_logistic(np.vstack([np.random.random((20,2)),1+np.random.random((20,2))]),np.hstack([np.tile(True,(20,)),np.tile(False,(20,))]))
first_part = np.random.random((20,2))
second_part = [1,0]+np.random.random((20,2))
first_vals = np.tile(True,(20,))
second_vals = np.tile(False,(20,))
plt.scatter(*first_part.T,color='green')
plt.scatter(*second_part.T,color='red')
def test_logistic(pts,normal):
    firsts = np.random.random((pts,2))
    seconds = normal + np.random.random((pts,2))
    trues = np.tile(True,(pts,))
def test_logistic(pts,normal):
    firsts = np.random.random((pts,2))
    seconds = normal + np.random.random((pts,2))
    trues = np.tile(True,(pts,))
    falses = np.tile(False,(pts,))
def test_logistic(pts,normal):
    firsts = np.random.random((pts,2))
    seconds = normal + np.random.random((pts,2))
    trues = np.tile(True,(pts,))
    falses = np.tile(False,(pts,))
    plt.scatter(*firsts.T,color='green')
    plt.scatter(*seconds.T,color='red')
    reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
    
test_logistic(100,[1,0])
def test_logistic(pts,normal):
    firsts = np.random.random((pts,2))
    seconds = normal + np.random.random((pts,2))
    trues = np.tile(True,(pts,))
    falses = np.tile(False,(pts,))
    plt.scatter(*firsts.T,color='green')
    plt.scatter(*seconds.T,color='red')
    result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
    return result    
test_logistic(100,[1,0])
test_logistic(100,[1,0])
test_logistic(100,[1,0])
test_logistic(100,[1,0])
np.linalg.norm(_)
__/_
np.linalg.norm(Out[135])
Out[135])/_
Out[135]/_
def test_logistic(pts,normal):
    firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
    seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
    trues = np.tile(True,(pts,))
    falses = np.tile(False,(pts,))
    plt.scatter(*firsts.T,color='green')
    plt.scatter(*seconds.T,color='red')
    result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
    return result
test_logistic(100,[1,0])
test_logistic(100,[0,1,0])
test_logistic(100,[1,0])
def test_logistic(pts,normal):
   firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
   seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
   trues = np.tile(True,(pts,))
   falses = np.tile(False,(pts,))
   plt.scatter(*firsts.T,color='green')
   plt.scatter(*seconds.T,color='red')
   result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
   return result
np.hstack([np.ones((10,1)),np.random.random((10,2))])
test_logistic(10,[1,0])
def test_logistic(pts,normal):
   firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
   seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
   trues = np.tile(True,(pts,))
   falses = np.tile(False,(pts,))
   plt.scatter(*firsts[:,1:].T,color='green')
   plt.scatter(*seconds[:,1:].T,color='red')
   result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
   return result
test_logistic(10,[1,0])
test_logistic(40,[1,0])
test_logistic(500,[1,0])
np.linalg.norm(_)
__/_
np.linalg.norm(Out[152][1:])
Out[152][1:]/_
def test_logistic(pts,normal):
   firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
   seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
   trues = np.tile(True,(pts,))
   falses = np.tile(False,(pts,))
   plt.scatter(*firsts[:,1:].T,color='green')
   plt.scatter(*seconds[:,1:].T,color='red')
   result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
   print(result)
   nrm = np.linalg.norm(result[1:])
   print("{0} + {1}".format(result[0],result[1:]/nrm))
   return result
test_logistic(500,[1,0])
def test_logistic(pts,normal):
   firsts = np.hstack([np.ones((pts,1)),np.random.random((pts,2))])
   seconds = np.hstack([np.ones((pts,1)),normal + np.random.random((pts,2))])
   trues = np.tile(True,(pts,))
   falses = np.tile(False,(pts,))
   plt.scatter(*firsts[:,1:].T,color='green')
   plt.scatter(*seconds[:,1:].T,color='red')
   result = reg.calibrate_logistic(np.vstack([firsts,seconds]),np.hstack([trues,falses]))
   print(result)
   nrm = np.linalg.norm(result[1:])
   print("{0} + {2}{1}".format(result[0],result[1:]/nrm,nrm))
   return result
test_logistic(500,[1,0])
test_logistic(5000,[1,0])
test_logistic(500000,[1,0])
test_logistic(50000,[1,0])
test_logistic(500,[1,0])
test_logistic(50,[1,0])
test_logistic(5,[1,0])