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
contactMatrixUSA_Base = pd.read_csv("./ContactMatrices/Base/ContactMatrixUSA_Base.csv").values
model.fitGraphToContactMatrix(contactMatrixUSA_Base, 'age', 5)
model.build(model.clusterStrogatz)
model.simulate(gamma, tau)
model.plotSIR()