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

graph = model.graph
nodes = graph.nodes()
#edges = graph.edges()
edges = list(graph.edges(data='transmission_weight'))
nodes = list(graph.nodes()

import numpy as np
import csv
np.savetxt("edges.csv", edges, delimiter=" ", fmt="%d %d %4.3f")
np.savetxt("nodes.csv", nodes, delimiter=" ", fmt="%d")

#model.infectPopulace(initial_infection_rate)

#simulate
#model.simulate(gamma,tau)
#model.plotSIR()
#
