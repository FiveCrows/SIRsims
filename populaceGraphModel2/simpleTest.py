from modelingToolkit import *
import json

# load default params for model
defaultParams  = (json.load(open('defaultParameters')))
globals().update(defaultParams)
#initialize model
model = PopulaceGraph(slim = False)

#initialize NetBuilder and build networks
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)
model.infectPopulace(init_infection_rate)

graph = model.graph
nodes = graph.nodes()
#edges = graph.edges()
edges = list(graph.edges(data='transmission_weight'))
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
np.savetxt("edges.csv", edges, delimiter=" ", fmt="%d %d %4.3f")
np.savetxt("nodes.csv", nodes_ages.transpose(), delimiter=" ", fmt="%d")

#model.infectPopulace(initial_infection_rate)

#simulate
#model.simulate(gamma,tau)
#model.plotSIR()
#
