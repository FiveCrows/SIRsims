from ContagionModeling import *
# plot infection chances
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :20}
default_env_masking = {'work': 0, 'school':0, 'household': 0}

gamma = 0.1
tau = 0.08

trans_weighter = TransmissionWeighter(default_env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, slim = False)
model.build(model.clusterStrogatz)
model.plotNodeDegreeHistogram()
