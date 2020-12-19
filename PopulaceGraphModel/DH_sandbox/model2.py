import networkx as nx
import EoN as eon
import numpy as np
import matplotlib.pyplot as plt

# Example SIR model using fast_SIR, based on an N to N graph (homogeneous connections. The graph is undirected. Transmission and recovery parameters are constant. 

beta = .02   # Transmission (S ->beta -> I) [1/days] (per edge)
gamma = .02  # Recovery (I -> R)  [1/days]  (per node)

# A graph with N nodes has N*(N-1)/2 edges. 

N = 100  # number of nodes
G = nx.complete_graph(N)
nb_nodes = G.number_of_nodes()

# Other useful graphs you could create
"""
G = nx.erdos_renyi_graph(n, p[, seed, directed])
G = nx.complete_bipartite_graph(3, 5)
G = nx.watts_strogatz_graph(n, k, p[, seed])
G = nx.barabasi_albert_graph(n, m[, seed])
"""


# Learn the dir() command
print(dir(G))

print("num nodes: ", G.number_of_nodes())
print("num edges: ", G.number_of_edges())

initial_recovered = []

# Single initial infected. Does not matter which one since the graph 
# is fully-connected
initial_infected = [1]

tmin, tmax = 0., 100.

sim_result = eon.fast_SIR(G, beta, gamma, initial_infecteds=initial_infected, 
        initial_recovereds=initial_recovered, 
        #transmission_weight="tw", 
        #recovery_weight="rw",
        return_full_data = True, 
        tmin=tmin, tmax=tmax)

# All the times for which there is data. Print out to see: floats. 
times = sim_result.t()
statuses = {}
last_time = times[-1]

# Specify the times you wish to work with. Tix is the time index. 
for tix in range(0, int(last_time)+2, 1):
    # statuses[tix]: for each node of the graph, S,I,R status at time tix
    # this is computed by some internal interpolation
    statuses[tix] = sim_result.get_statuses(time=tix)
    # print("time= ", tix, "  status= ", statuses[tix])

print(statuses[0])

print(sum(1 for x in statuses[2].values() if x == 'I'))

# statuses.keys() tells you the times at which data is saved.
# statuses[2] is a dictionary of the network at t=2., for example: 
# status[2] = {0: 'I', 1: 'I', 2: 'I', 3: 'R', 4: 'R', 5: 'R', 6: 'R', 7: 'R', 8: 'I', 9: 'I', 10: 'I', 11: 'I', 12: 'I', 13: 'R', 14: 'I', 15: 'I', 16: 'R', 17: 'R', 18: 'R', 19: 'R', 20: 'R', 21: 'I', 22: 'I', 23: 'I', 24: 'R', 25: 'R', 26: 'R', 27: 'R', 28: 'R', 29: 'I'}

# Together with lat/long for each node, you can plot the map. 

# Synthetic lat/long (random chocie among [0,100])
x = np.random.choice(100, nb_nodes, replace=False)
y = np.random.choice(100, nb_nodes, replace=False)

# Replace: 'S', 'I', 'R' by 0, 1, 2 in statuse

subst_dict = {'S':0,  'I':1, 'R':2}
print("keys: ", statuses[1].keys())
print("last_time= ", last_time)

def replace(time_index):
    d = np.zeros(len(statuses[0]), dtype='int')
    for k,v in statuses[time_index].items():
        d[k] = subst_dict[v]
    return d


quit()
#for i in range(0, int(last_time), 2):
#    print(replace(i))

# Notes: https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
nrows = 6
ncols = 6
plt.subplots(ncols,nrows)

import matplotlib as mpl
# Map 0,1,2 to black, red, green
cmap = mpl.colors.ListedColormap(['black', 'red', 'green'])
c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=4)


for i in range(nrows*ncols):
    plt.subplot(ncols,nrows,i+1)
    try:
        col = replace(2*i)
    except:
        break
    plt.scatter(x, y, c=col, s=3*3*3, cmap=cmap, norm=c_norm, alpha=0.5)
plt.colorbar()
plt.show()
