from ge_modelingToolkit import *
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
# https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_SIR.html
# Argument to EoN.fast_SIR(G, tau, gamma, initial_infecteds=None,
gamma                 = 0.2  # Recovery rate per edge (EoN.fast_SIR)
tau                   = 0.2  # Transmission rate per node (EoN.fast_SIR) (also called beta in the literature)

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7254020/
#  Provided by Derek. No sueful data. 
# https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf#:~:text=Using%20available%20preliminary%20data%2C,severe%20or%20critical%20disease.
#   - gives average rcovery rates
# https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
#   - paper has tables of parameters sometimes stratified by age
# https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1003346&type=printable
#    Tons of tables of studies, but use is not clear. 
# https://f1000researchdata.s3.amazonaws.com/manuscripts/28897/a4d3438e-54c8-4f2f-bdbe-cd913f104085_26186_-_mudatsir.pdf?doi=10.12688/f1000research.26186.1&numberOfBrowsableCollections=27&numberOfBrowsableInstitutionalCollections=5&numberOfBrowsableGateways=26 (Predictors of COVID-19 severity: a systematic review and meta-analysis [version 1; peer review: 1 approved]
# https://pubmed.ncbi.nlm.nih.gov/32497510/ (Physical distancing, face masks, and eye protection to prevent person-to-person transmission of SARS-CoV-2 and COVID-19: a systematic review and meta-analysis)
#   Has to do with masks. 

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
model          = PopulaceGraph( partition, slim = False)

# Create Graph
model.build(trans_weighter, preventions, env_degrees)
# Run SIR simulation
model.simulate(tau, gamma, title = 'base-test')

school_masks = copy.deepcopy(preventions)
school_masks['school']['masking'] = 1

pass

with_distancing = copy.deepcopy(preventions)
with_distancing['workplace']['distancing'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(tau, gamma, title = 'school and workplace distancing')

env_degrees['school'] = 0
model.reweight(trans_weighter, with_distancing)
model.simulate(tau, gamma, title = 'schools closed')

preventions['workplace']['masking'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(tau, gamma, title = 'schools closed, and workplaces masked')


#-----------------------------------------
# Simulations ended
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
