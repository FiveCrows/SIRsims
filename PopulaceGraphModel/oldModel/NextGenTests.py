from oldModel.ContagionModeling import *
from matplotlib import pyplot as plt
from numpy import linalg as LA

mask_scalar = 0.3
env_scalars = {"school": 0.3 , "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :13}
default_env_masking = {'work': 0, 'school':0, 'household': 0}
gamma = 0.1
tau = 0.08

trans_weighter = TransmissionWeighter(env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, slim = True)

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
model.build(model.clusterStrogatz)
model.simulate(gamma, tau, title = 'schools open, no masks')
model.getR0()
partition,id_to_partition = model.partition(list(model.graph.nodes),'age', enumerator)

N = model.returnNextGenMatrix(partition, id_to_partition,tau,gamma)
print(LA.eig(N))
plt.show()
print("total edge weight:{}".format( model.sumAllWeights()))
model.simulate(gamma, tau, title = 'schools open, no masks')

model.build(model.clusterStrogatz)
print("total edge weight:{}".format( model.sumAllWeights()))
model.simulate(gamma, tau, title = 'schools closed')