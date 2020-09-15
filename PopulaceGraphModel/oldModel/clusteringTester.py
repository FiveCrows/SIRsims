from oldModel.ContagionModeling import *
from matplotlib import pyplot as plt
import networkx as nx

mask_scalar = 0.3
env_scalars = {"school": 0.3 , "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :13}
default_env_masking = {'work': 0, 'school':0, 'household': 0}
gamma = 0.1
tau = 0.08


trans_weighter = TransmissionWeighter(env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, slim = True)
Anodes = list(range(10))
Bnodes = list(range(10,20))
model.clusterBipartite(Anodes, Bnodes,30,1, p_random = 0.2)
pos = nx.spring_layout(model.graph)
nx.draw_networkx_nodes(model.graph, pos, nodelist = Anodes, node_color = 'r')
nx.draw_networkx_nodes(model.graph, pos, nodelist = Bnodes, node_color = 'b')
nx.draw_networkx_edges(model.graph, pos)
plt.show()