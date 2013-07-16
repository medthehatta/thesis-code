import os.path
import matplotlib.pyplot as plt
import numpy as np
import log_regression as lr
import stage

def plot_classifier(cal,trues,falses):
    x = np.linspace(-1,1,200)
    y = np.linspace(-1,1,200)
    (X,Y) = np.meshgrid(x,y)
    XY = np.dstack([X,Y])
    Z = lr.evaluate_poly(cal,lr.dim2_deg4[:,None,None,:],XY)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.grid(True)
    ax.axis('equal')

    ax.contour(X,Y,Z,levels=np.linspace(np.min(Z),np.max(Z),20),colors='k')

    # adjust for the way the random samples were distributed
    adjustment = 1

    if len(falses)>0:
        falses = np.diagonal(falses,axis1=1,axis2=2)[:,:2] - [adjustment]*2
        ax.scatter(*falses.T,color='red')

    if len(trues)>0:
        trues = np.diagonal(trues,axis1=1,axis2=2)[:,:2] - [adjustment]*2
        ax.scatter(*trues.T,color='green')

    return f

def plot_classifier_here(cal,trues,falses,title=""):
    fig = plot_classifier(cal,trues,falses)
    fig.suptitle(title)
    PFX = "/home/med/astro/stuff/test_plots"
    file = "C_{}.png".format(np.random.randint(999))
    fig.savefig(os.path.join(PFX,file))
    print("http://astro.temple.edu/~tud48344/stuff/test_plots/{}".format(file))

