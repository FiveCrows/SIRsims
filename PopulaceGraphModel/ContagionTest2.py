from ContagionModeling import *

mask_scalar = 0.3
env_scalars = {"school": 0.3 , "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :20}
default_env_masking = {'work': 0, 'school':0, 'household': 0}
gamma = 0.1
tau = 0.08

trans_weighter = TransmissionWeighter(env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, slim = False)
trans_weighter.env_masking = {'work': 1, 'school': 1, 'household': 0}
model.build(model.clusterStrogatz)
print("total edge weight:{}".format( model.sumAllWeights()))
model.simulate(gamma, tau, title = 'school and work masks')
trans_weighter.env_masking = default_env_masking
trans_weighter.env_scalars = {"school": 0, "work": 0.3, "household": 1}
model.build(model.clusterStrogatz)
print("total edge weight:{}".format( model.sumAllWeights()))
model.simulate(gamma, tau, title = 'schools closed')
trans_weighter.env_scalars = env_scalars
model.build(model.clusterStrogatz)
print("total edge weight:{}".format( model.sumAllWeights()))
model.simulate(gamma, tau, title = 'schools open, no masks')
model.plotSIR()

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
partition,id_to_partition = model.partition(list(model.graph.nodes),'age', enumerator)
model.plotEvasionChart(partition)