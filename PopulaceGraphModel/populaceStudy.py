from ModelToolkit import *
# plot chance of infection
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 13, 'school': 13}
default_env_masking = {'workplace': 0, 'school':0, 'household': 0}
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
partition = Partitioner(enumerator, 'age', names)

model = PopulaceGraph( partition, slim = False)
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')

school_masks = dict(preventions)
school_masks['school']['masking'] = 1
model.build(trans_weighter, school_masks, env_degrees)
model.simulate(gamma, tau, title = 'in-school masks')
pass

with_distancing = dict(preventions)
with_distancing['workplace']['distancing'] = 1
model.build(trans_weighter, with_distancing, env_degrees)
model.simulate(gamma, tau, title = 'school and workplace distancing')



env_degrees['school'] = 0
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'schools closed')

preventions['workplace']['masking'] = 1
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'schools closed, and workplaces masked')

globalMultiEnvironment = model.returnMultiEnvironment(model.environments.keys(), partition)
largestWorkplace = model.environments[505001334]
largestSchool = model.environments[450059802]
#bigHousehold = model.environments[58758613]
list = [globalMultiEnvironment, largestWorkplace, largestSchool]#, #bigHousehold]
for environment in list:
    model.plotBars(environment)#(environment = model.environments[505001334])
    model.plotNodeDegreeHistogram(environment)
    if environment != globalMultiEnvironment:
        model.plotContactMatrix(environment)
model.plotSIR()