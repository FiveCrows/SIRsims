from ModelToolkit import *
import copy
# plot chance of infection

#TODO Upgrade the paper
#TODO run a range for default_env_scalars, and plot ???
#TODO recreate model object for multiple sims
#TODO test contact matrix sensitivities
#TODO

#These values scale the weight that goes onto edges by the environment type involved
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1} # base case
#As None, the degrees of the environment are implicit in contact matrices
env_degrees = {'workplace': None, 'school': None}
#the prevention measures in the workplaces
workplace_preventions = {'masking': 0.0, 'distancing': 0}
#the prevention measures in the schools
school_preventions = {'masking':0.0, 'distancing': 0}
#the prevention measures in the household
household_preventions = {'masking':0.0, 'distancing':0}
#combine all preventions into one var to easily pass during reweight and build
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
#these values specify how much of a reduction in weight occurs when people are masked, or distancing
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values

#this object holds rules and variables for choosing weights
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
#gamma is the recovery rate, and the inverse of expected recovery time
gamma = 0.1
#tau is the transmission rate
tau = 0.08

#the partioner is needed to put the members of each environment into a partition,
#currently, it is setup to match the partition that is implicit to the loaded contact matrices
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner('age', enumerator, names)

#----------- BEGIN SIMULATIONS ------------------------------------

#init, build simulate
model = PopulaceGraph( partition, slim = False)
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')

#new parameter sets for different tests
school_masks = copy.deepcopy(preventions)
school_masks['school']['masking'] = 1
with_distancing = copy.deepcopy(preventions)
with_distancing['workplace']['distancing'] = 1

#weight with distancing applied to workplace and schools, and resimulate
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'school and workplace distancing')

#close schools
default_env_scalars['school'] = 0
trans_weigher.setEnvScalars(default_env_scalars)
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed')

#I change the default preventions here to use but it doesn't matter because this is the last one
preventions['workplace']['masking'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed, and workplaces masked')

#----------- END SIMULATIONS ---------------------------------------

#the globalMultiEnvironment is useful if one wants to make the all of leon country
globalMultiEnvironment = model.returnMultiEnvironment(model.environments.keys(), partition)
#some interesting environment to investigate the results in, others in usefulEnvironments.txt
largestWorkplace = model.environments[505001334]
largestSchool = model.environments[450059802]
#bigHousehold = model.environments[58758613]

#this will plot S, I and R curves for the entire populace
model.plotSIR()

#by calling with largestWorkplace, plot will be made for that specific environment
model.plotNodeDegreeHistogram(largestWorkplace)
model.plotContactMatrix(largestWorkplace)
#bug seems to keep coming back in the contactmatrix generator, though I've fixed it, rip, wip...
plt.imshow(largestWorkplace.contact_matrix)

#priority
#Show charts for bipartite n1,n2,m1,m2
#add plots to overleaf
#add description to overleaf

#plot some network-charts

