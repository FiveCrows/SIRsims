from ModelToolkit import *
# plot chance of infection
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 13, 'school': 13}
default_env_masking = {'workplace': 0, 'school':0, 'household': 0}
workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions = {'masking':0, 'distancing': 0}
household_preventions = {}
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
trans_weighter = TransmissionWeighter(default_env_scalars, mask_scalar, default_env_masking)

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partition(enumerator, 'age', names)

model = PopulaceGraph(trans_weighter, partition, slim = False)
model.build(preventions, env_degrees)
model.plotNodeDegreeHistogram()
