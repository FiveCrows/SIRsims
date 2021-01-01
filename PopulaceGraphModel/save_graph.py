from ge_modelingToolkit2 import *
import json

script_file = "ge_base_study.py"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
slim = False

# load default params for model
defaultParams  = (json.load(open('defaultParameters')))
globals().update(defaultParams)
#initialize model
prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}  
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)

#model = PopulaceGraph(slim = False)
model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim, timestamp=timestamp)

#initialize NetBuilder and build networks
#netBuilder = NetBuilder()
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies)
#model.networkEnvs(netBuilder)

infect_perc = 0.001
model.infectPopulace(perc=infect_perc)
# Bryan called weights before completing the network. Illogical
#model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)
model.infectPopulace(init_infection_rate)
model.buildNetworks(netBuilder)

graph = model.graph
print("Graph stats")
print(graph.number_of_nodes(), graph.number_of_edges())

graph = model.graph
nodes = graph.nodes()
#edges = graph.edges()
#edges = list(graph.edges(data='transmission_weight'))
edges = np.asarray(list(graph.edges))
nodes = list(graph.nodes())

ageGroups = [[0,5], [5,18], [18,50], [50,65], [65,100]]
ages = []
for node in model.populace:
    ages.append(node['age'])

enumerator = {}
enumerator = np.zeros(len(model.populace), dtype='int')
for i, group in enumerate(ageGroups):
    enumerator[group[0]:group[1]] = i 

groups = enumerator[ages]

nodes_ages = np.vstack([nodes, groups])

import numpy as np
import csv
print("edges= ", edges[0])
print("edges= ", edges[1])
weights = np.ones([edges.shape[0], 1])
# Add weight of 1
edges = np.hstack([edges, weights])

with open("edges.csv", "w") as fd:
    fd.write("%d\n" %graph.number_of_edges())
    np.savetxt(fd, edges, delimiter=" ", fmt="%d %d %4.3f")

with open("nodes.csv", "w") as fd:
    fd.write("%d\n" %graph.number_of_nodes())
    np.savetxt(fd, nodes_ages.transpose(), delimiter=" ", fmt="%d")

