from ContagionModeling import *
import networkx as nx
import random as random
#mask_scalar scales down the weights between nodes with mask use
mask_scalar = 0.3
#env_scalars scales weights to the environment in which people interact
env_weights = {"school": 0.3 , "work": 0.3, "household": 1}
#env_degrees specifies how many edges are connected for people at an environment, assuming
#there are at least enough people to match the degree, otherwise it's just fully connected
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0, 'household':0}
gamma = 0.1
tau = 0.08

#the transmission weighter defines how weights are chosen on each edge
weighter = TransmissionWeighter(env_weights, mask_scalar, env_masking)
model = PopulaceGraph(weighter, env_degrees, slim = False)
model.build(model.clusterStrogatz)
model.simulate(gamma, tau, title = 'With environment clustering topology')

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
partition,id_to_partition = model.partition(list(model.graph.nodes),'age', enumerator)

replGraph = nx.Graph()
weights = nx.get_edge_attributes(model.graph, 'transmission_weight')

for edge in model.graph.edges():
    i = id_to_partition[edge[0]]
    j = id_to_partition[edge[1]]
    weight = weights[edge]
    edgeReplaced = False
    while not edgeReplaced:
        A = random.choice(partition[i])
        B = random.choice(partition[j])
        if not replGraph.has_edge(A,B):
            replGraph.add_edge(A,B, transmission_weight = weight)
            edgeReplaced = True
model.graph = replGraph
model.simulate(gamma, tau, title ='with random topology')
model.plotSIR()
#model.build builds the models graph. I call it with clusterStrogatz as the clustering algorithm here,
#to specify that edges at work and schools are picked for strogatz graphs
