import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

from ge_modelingToolkit2 import *
import numpy as np

# Author: Gordon Erlebacher
# Date: 2020-11-18
# Generate graph degree histograms for workplace and schools
# Questions: how to handle workplaces of 1 person? 

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


model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim, timestamp=timestamp)
#netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, cv_dict={})
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies) #, cv_dict={})
model.buildNetworks(netBuilder)

h_sizes = [0]*13
s_sizes = [0]*35
w_sizes = [0]*10

#-------------------------------------
def edgeHistogram(edge_list):
    # compute degree of each node
    degree = defaultdict(int)
    hist = defaultdict(int)
    for e in edge_list:
        degree[e[0]] += 1
        degree[e[1]] += 1

    # compute histogram of graph
    for node, deg in degree.items():
        hist[deg] += 1
    
    return hist 
#-------------------------------------

envs = {'household':[], 'school':[], 'workplace':[]}
envs_hist = {'household':[], 'school':[], 'workplace':[], 'all':[]}

for index in model.environments:
    env = model.environments[index]
    envs[env.env_type].extend(env.members)
    hist = edgeHistogram(env.edges)
    envs_hist[env.env_type].append(hist)
    envs_hist['all'].append(hist)

def avgHistogram(envs):
    nb_hist = len(envs)
    avg_hist = defaultdict(int)
    for i in range(nb_hist):
        hist = envs[i]
        for k,v in hist.items():
            avg_hist[k] += v

    nb_nodes = sum(avg_hist.values())
    #for k,v in avg_hist.items():
        #avg_hist[k] /= nb_hist
    # normalize by the total number of nodes
    for k in avg_hist:
        avg_hist[k] /= nb_nodes
    return avg_hist

sch_hist = avgHistogram(envs_hist['school'])
print("school: ", sch_hist)
wrk_hist = avgHistogram(envs_hist['workplace'])
print("workplace: ", wrk_hist)
hm_hist = avgHistogram(envs_hist['household'])
print("household: ", wrk_hist)
all_hist = avgHistogram(envs_hist['all'])  # full graph
print("all: ", wrk_hist)

k_sch = list(sch_hist.keys())
k_wrk = list(wrk_hist.keys())
k_hm = list(hm_hist.keys())
k_all = list(all_hist.keys())
print("k_sch= ", k_sch)
print("k_wrk= ", k_wrk)
print("k_hm= ", k_hm)
print("k_all= ", k_all)

max_deg = max(k_sch + k_wrk + k_hm)
degrees = list(range(0,max_deg))
deg_sch = []
deg_wrk = []
deg_hm  = []
deg_all  = []
for d in degrees:
    deg_sch.append(sch_hist[d])
    deg_wrk.append(wrk_hist[d])
    deg_hm.append(hm_hist[d])
    deg_all.append(all_hist[d])
df = pd.DataFrame({'degree': degrees, 'school': deg_sch, 'workplace':deg_wrk, 'household':deg_hm, 'all':deg_all})
print(df)

sns.barplot(x='degree', y='workplace', data=df, color='red', alpha=0.5, label='Workplace') #palette="Reds")
sns.barplot(x='degree', y='school', data=df, color='blue', alpha=0.5, edgecolor='blue', label='School') #palette="Blues", alpha=0.7)
ax = sns.barplot(x='degree', y='household', data=df, lw=0, color='green', alpha=0.5, edgecolor='green', label='Household') #palette="Reds")

ax.set_xticks(np.linspace(0, 40, 21, dtype='int'))
#ax.set_xlim(0, 36)
plt.ylabel('Percentage')
plt.title("Histogram of home, workplace, and school node degree")
plt.legend()
plt.savefig("plot_degree_histograms.pdf")

# plot of degree for full graph
#df = pd.DataFrame({'degree': degrees, 'all': deg_all, 'workplace':deg_wrk, 'household':deg_hm})
ax = sns.barplot(x='degree', y='all', data=df, color='black', alpha=0.2, label='Full Graph') #palette="Reds")

lw = 1.5
ax = sns.barplot(x='degree', y='school', data=df, lw=lw, facecolor=(1,1,1,0), edgecolor='blue') #palette="Reds")
ax = sns.barplot(x='degree', y='workplace', data=df, lw=lw, facecolor=(1,1,1,0), edgecolor='red') #palette="Reds")
ax = sns.barplot(x='degree', y='household', data=df, lw=lw, facecolor=(1,1,1,0), edgecolor='green') #palette="Reds")
ax = sns.barplot(x='degree', y='all', data=df, lw=lw, facecolor=(1,1,1,0), edgecolor='black') #palette="Reds")

ax.set_xticks(np.linspace(0, 40, 21, dtype='int'))
plt.legend()
plt.savefig("plot_degree_histograms_fullgraph.pdf")

quit()


