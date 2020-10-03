from ge_ModelToolkit import *
import copy
# plot chance of infection

#################################################################################
#####   Begin setting up model variables  #######################################
#################################################################################

default_env_scalars   = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees           = {'workplace': None, 'school': None}
default_env_masking   = {'workplace': 0, 'school':0, 'household': 0}
workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions    = {'masking':0, 'distancing': 0}
household_preventions = {'masking':0, 'distancing':0}
preventions           = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values
gamma                 = 0.1   # Per edge transmission or recovery? 
tau                   = 0.08  # Per edge transmission or recovery? 
names                 = ["{}:{}".format(5 * i, 5 * (i + 1)-1) for i in range(15)]
names.append("75-100")
print("Age brackets: ", names)

# Dictionary of ages: enumerator[i] => age bracket[i]
# Age brackets: range(0,5), range(5:10), ..., range(70:75), range(75:100)
enumerator            = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
partition      = Partitioner('age', enumerator, names)
model          = PopulaceGraph( partition, slim = True)

# Create Graph
model.build(trans_weighter, preventions, env_degrees)
# Run SIR simulation
model.simulate(gamma, tau, title = 'base-test')

school_masks = copy.deepcopy(preventions)
school_masks['school']['masking'] = 1
model.build(trans_weighter, school_masks, env_degrees)
model.simulate(gamma, tau, title = 'in-school masks')
pass

with_distancing = copy.deepcopy(preventions)
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

model.plotSIR()
model.plotNodeDegreeHistogram(largestWorkplace)
model.plotContactMatrix(largestWorkplace)
plt.imshow(largestWorkplace.contact_matrix)

#priority
#Show charts for bipartite n1,n2,m1,m2
#add plots to overleaf
#add description to overleaf

#plot some network-charts
