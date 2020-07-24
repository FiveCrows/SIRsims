from ContagionModeling import *
import pandas as pd
#mask_scalar scales down the weights between nodes with mask use
mask_scalar = 0.3
#env_weights scales weights to the environment in which people interact
env_weights = {"school_id": 0.1 , "work_id": 0.2, "sp_hh_id": 1}
#env_degrees specifies how many edges are connected for people at an environment, assuming
#there are at least enough people to match the degree, otherwise it's just fully connected
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0}
gamma = 0.1
tau = 0.08

#the transmission weighter defines how weights are chosen on each edge
weighter = TransmissionWeighter(env_weights, mask_scalar)
model = PopulaceGraph(weighter, env_degrees, env_masking)

#model.build builds the models graph. I call it with clusterStrogatz as the clustering algorithm here,
#to specify that edges at work and schools are picked for strogatz graphs
model.build(model.clusterStrogatz)
contactMatrixUSA_Base = pd.read_csv("./ContactMatrices/Base/ContactMatrixUSA_Base.csv").values
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