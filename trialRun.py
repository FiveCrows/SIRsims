from ContagionModeling import *
import pandas as pd

mask_scalar = 0.3
env_weights = {"school_id": 0.1 , "work_id": 0.2, "sp_hh_id": 1}
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0}
gamma = 0.1
tau = 0.08
weighter = TransmissionWeighter(env_weights, mask_scalar)
model = PopulaceGraph(weighter, env_degrees, env_masking)
model.build(model.clusterStrogatz)
contactMatrixUSA_Base = pd.read_csv("./ContactMatrices/Leon/ContactMatrixLeonAll.csv", header=None).values
model.simulate(gamma, tau, title = 'control')
#model.plotContactMatrix()
model.fitWithContactMatrix(contactMatrixUSA_Base, 'age', 5, show_scale = True)
#model.plotVictoryChart()
model.simulate(gamma, tau, title = 'fit to Contact Matrix')
#model.plotVictoryChart()
model.plotSIR()
#model.plotNodeDegreeHistogram()
