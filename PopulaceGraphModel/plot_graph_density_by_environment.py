import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

from ge_modelingToolkit2 import *
import numpy as np

# Author: Gordon Erlebacher
# Date: 2020-11-18
# Generate graph density degree histograms for workplace and schools
# Density is the ratio of the number of edges divided by the max possible nb edges
# Not sure what that tells us. 

save_output = False
slim = True
slim = False

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)-1) for i in range(15)]
names.append("{}:{}".format(75,100))
partitioner = Partitioner('age', enumerator, names)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}  
# denotes the fraction of people using masks
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}

env_type_scalars = {"household": 1, "school": 1.0, "workplace": 1.0}

glob_dict = {}
glob_dict["gamma_shape"] = 1.
glob_dict["gamma_scale"] = 1.
gamma = 1.
tau = 1.

model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim, timestamp=timestamp)
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, cv_dict={})
model.buildNetworks(netBuilder)

#-------------------------------------
def densityHistogram(edges, population):
    max_nb_edges = population * (population-1) / 2
    nb_edges = len(edges)
    if max_nb_edges == 0: max_nb_edges = 1
    density = nb_edges / max_nb_edges
    return density
#-------------------------------------

envs = {'household':[], 'school':[], 'workplace':[]}

for index in model.environments:
    env = model.environments[index]
    if env.population > 4:
        density = densityHistogram(env.edges, env.population)
        envs[env.env_type].append(density)

print(envs['workplace'])
print(envs['school'])

sns.set_theme()
fig, axes = plt.subplots(1,2)
fig.suptitle("Density of Environment Graphs")
plt.subplot(1,2,1)
sns.distplot(envs['school'], kde_kws={"clip":(0,1)})
plt.title("School")
plt.subplot(1,2,2)
sns.distplot(envs['workplace'], kde_kws={"clip":(0,1)})
plt.title("Workplace")
plt.tight_layout()
plt.savefig("plot_graph_density_by_environment.pdf")
quit()

