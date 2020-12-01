import random
from os import mkdir
import EoN
import networkx as nx
import itertools
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.stats import bernoulli

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3,3)
axes = axes.reshape(-1)
shapes = [0.85, 1.46, .2]
scales = [4.35, 3.57, 2.]

shapes = [5., 0.2, .4]
scales = [0.1, 10., 2.]
shapes = [5.]
scales = [.1,.5,1.,2.,.5,10.,50.,100, 500.]

fig.suptitle("$\Gamma$(shape, scale)")
for ish, shape in enumerate(shapes):
    for isc, scale in enumerate(scales):
        mean = shape
        var  = mean*scale
        gamma = np.random.gamma(shape/scale, scale, 10000)
        npvar = np.var(gamma)
        print("---- a, b= ", shape/scale, scale)
        print("$\mu$($\gamma$)= ", np.mean(gamma))
        print("$\sigma$= ", np.sqrt(np.var(gamma)))
        print("var= ", var, ", npvar= ", npvar)
        npvar = np.var(gamma)
        print("dispersion: ", var / mean**2)
        dispersion = var / mean**2
        #ax = axes[ish, isc]
        ax = axes[isc]
        plt.sca(ax)
        print("len(gamma)= ", len(gamma))
        sns.histplot(gamma, bins=300)
        #sns.histplot(gamma, kde_kws={"clip":(0,3)}, bins=300)
        print("after histplot")
        #ax.set_title(r"$\Gamma$(%3.2f, %3.2f)" % (shape, scale))
        ax.set_title(r"$\Gamma$(%3.2f, %3.2f), $\mu$=%3.2f, k=%3.2f, $\sigma$^2=%3.2f" % (shape*scale, scale, mean, dispersion, npvar), fontsize=4)
        ax.set_xlim(0,20)

plt.tight_layout()
plt.savefig("plot_gamma_distribution.pdf")
