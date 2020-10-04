from ModelToolkit import *
import copy
# plot chance of infection

#TODO Upgrade the paper
#TODO run a range for default_env_scalars, and plot ???
#TODO recreate model object for multiple sims
#TODO test contact matrix sensitivities
#TODO
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1} # base case
env_degrees = {'workplace': None, 'school': None}
workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions = {'masking':0, 'distancing': 0}
household_preventions = {'masking':0, 'distancing':0}
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
#gamma is the recovery rate, and the inverse of expected recovery time
gamma = 0.1
#tau is the transmission rate
tau = 0.08
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner('age', enumerator, names)
model = PopulaceGraph( partition, slim = True)

model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')

school_masks = copy.deepcopy(preventions)
school_masks['school']['masking'] = 1

pass

with_distancing = copy.deepcopy(preventions)
with_distancing['workplace']['distancing'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'school and workplace distancing')

env_degrees['school'] = 0
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed')

preventions['workplace']['masking'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed, and workplaces masked')

globalMultiEnvironment = model.returnMultiEnvironment(model.environments.keys(), partition)
largestWorkplace = model.environments[505001334]
largestSchool = model.environments[450059802]
#bigHousehold = model.environments[58758613]

model.plotSIR()
model.plotNodeDegreeHistogram(largestWorkplace)
model.plotContactMatrix(largestWorkplace)
plt.imshow(largestWorkplace.contact_matrix)

#priority
#Show charts for bipartite n1,n2,m1,m2
#add plots to overleaf
#add description to overleaf

#plot some network-charts
