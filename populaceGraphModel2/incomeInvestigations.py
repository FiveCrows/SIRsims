##################################################################
#This script is here to reveal information about the incomes of the population
##################################################################
import matplotlib as plt
from modelingToolkit import *
import resultAnalysisToolkit as rat
import json
import seaborn as sns
import partitioners as parts
# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = False)
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

#get and plot a distribution of incomes for each household 
envs = model.environments
households = rat.filterEnvByType(envs,'household')

#plot a distribution of incomes

#hh_incomes = [hh.income for hh in households.values()]
#getIncome = lambda person: households[person.sp_hh_id].income
#personal_incomes = [households[pers['sp_hh_id']].income for pers in model.populace]
#sns.distplot(hh_incomes, kde = True, axlabel = 'income',label = 'household_incomes')
#sns.distplot(personal_incomes, kde = True, axlabel = 'person_incomes')
#plt.show()

#create a partitioner to group people by income. 
ip = parts.autoBinLambdaPartitioner.incomePartitioner(model.environments, 10)
ip.partitionGroupWithAutoBound(model.populace)
rat.plotContactMatrix(model, ip, model.environments.keys())

