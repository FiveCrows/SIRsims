from ModelToolkit import *
import copy
# plot chance of infection
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 15, 'school': 13}
#default_env_masking = {'workplace': 0.3, 'school':0.3, 'household': 0}
workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions = {'masking':0, 'distancing': 0}
household_preventions = {'masking':0, 'distancing':0}
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
gamma = 0.1
tau = 0.08

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner('age', enumerator, names)
model = PopulaceGraph( partition, slim = True)
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')
largestWorkplace = model.environments[505001334]
model.plotContactMatrix(largestWorkplace)
plt.imshow(largestWorkplace.returnReciprocatedCM())
plt.show()