from oldModel.ContagionModeling import *
import pandas as pd
#mask_scalar scales down the weights between nodes with mask use
mask_scalar = 0.3
#env_scalars scales weights to the environment in which people interact
env_weights = {"school": 0 , "work": 0, "household": 1}
#env_degrees specifies how many edges are connected for people at an environment, assuming
#there are at least enough people to match the degree, otherwise it's just fully connected
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0, 'household':0}
gamma = 0.1
tau = 0.08

#the transmission weighter defines how weights are chosen on each edge
weighter = TransmissionWeighter(env_weights, mask_scalar, env_masking)
model = PopulaceGraph(weighter, env_degrees, slim = True)
model.build(model.clusterStrogatz)
#model.build builds the models graph. I call it with clusterStrogatz as the clustering algorithm here,
#to specify that edges at work and schools are picked for strogatz graphs

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
partition,id_to_partition = model.partition(list(model.graph.nodes),'age', enumerator)
preferenceMatrix = model.partitionToPreferenceMatrix(partition,id_to_partition)
contactMatrix = model.partitionToContactMatrix(partition, id_to_partition)
plt.imshow(preferenceMatrix)
plt.imshow(contactMatrix)
contactMatrixUSA_Base = pd.read_csv("../../ContactMatrices/Leon/ContactMatrixLeonAll.csv", header = None).values
#model.simulate
model.simulate(gamma, tau, title = 'control')
#model.plotContactMatrix()
model.fitWithContactMatrix(contactMatrixUSA_Base, 'age', 5, show_scale = True)
model.simulate(gamma, tau, title = 'fit to Contact Matrix')
model.plotBars()
model.plotSIR()
model.plotNodeDegreeHistogram()
model.record.dump()