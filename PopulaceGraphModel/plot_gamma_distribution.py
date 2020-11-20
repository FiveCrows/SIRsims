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

shape, scale = .1, 10.

fig, axes = plt.subplots(3,3)
shapes = [.1, .5, 2.]
scales = [.1, .5, 2.]

fig.suptitle("$\Gamma$(shape, scale)")
for ish, shape in enumerate(shapes):
    for isc, scale in enumerate(scales):
        gamma = np.random.gamma(shape, scale, 10000)
        ax = axes[ish, isc]
        plt.sca(ax)
        sns.distplot(gamma, kde_kws={"clip":(0,100)})
        ax.set_title(r"$\Gamma$(%3.2f, %3.2f)" % (shape, scale))

plt.tight_layout()
plt.savefig("plot_gamma_distribution.pdf")
