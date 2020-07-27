from ContagionModeling import *
import pandas as pd
#mask_scalar scales down the weights between nodes with mask use
mask_scalar = 0.3
#env_weights scales weights to the environment in which people interact
env_weights = {"school_id": 0 , "work_id": 0, "sp_hh_id": 1}
#env_degrees specifies how many edges are connected for people at an environment, assuming
#there are at least enough people to match the degree, otherwise it's just fully connected
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0}
gamma = 0.1
tau = 0.08

#the transmission weighter defines how weights are chosen on each edge
weighter = TransmissionWeighter(env_weights, mask_scalar)
model = PopulaceGraph(weighter, env_degrees, env_masking, slim = True)

#model.build builds the models graph. I call it with clusterStrogatz as the clustering algorithm here,
#to specify that edges at work and schools are picked for strogatz graphs
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
model.build(model.clusterStrogatz)
partition,id_to_partition = model.partition(list(model.graph.nodes),'age', enumerator)

preferenceMatrix = model.partitionToPreferenceMatrix(partition,id_to_partition)
plt.imshow(preferenceMatrix)
plt.show
contactMatrixUSA_Base = pd.read_csv("../ContactMatrices/Leon/ContactMatrixLeonAll.csv").values
#model.simulate
model.simulate(gamma, tau, title = 'control')
model.record.dump()
#model.plotContactMatrix()
model.fitWithContactMatrix(contactMatrixUSA_Base, 'age', 5, show_scale = True)
model.simulate(gamma, tau, title = 'fit to Contact Matrix')

model.plotEvasionChart()
model.plotSIR()
model.plotNodeDegreeHistogram()
model.record.dump()