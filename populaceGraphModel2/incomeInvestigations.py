##################################################################
#This script is here to reveal information about income
# contact relations of the population
##################################################################
import matplotlib as plt
from modelingToolkit import *
import resultAnalysisToolkit as rat
import graphAnalysisToolkit as mat
import numpy as np 
import json
import seaborn as sns
import partitioners as parts
# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = False)
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)

#construct ip, a partitioner by income 
num_income_brackets = 20
ip = parts.autoBinLambdaPartitioner.incomePartitioner(model.environments, num_income_brackets)
school_children = list(filter(lambda x: not np.isnan(x.school_id), model.populace))
employed = list(filter(lambda x: not np.isnan(x.work_id), model.populace))
school_partition = ip.partitionGroupWithAutoBound(school_children)
employed_partition = ip.partitionGroupWithAutoBound(employed)
full_partition = ip.partitionGroupWithAutoBound(list(model.populace))
#produces partitition bounds

#plot Contact between income groups for different env types

schools = model.filterEnvironments(lambda env: env.env_type == 'school')
workplaces = model.filterEnvironments(lambda env: env.env_type == 'workplace')
households = model.filterEnvironments(lambda env: env.env_type == 'household')
#fig, ax = plt.subplots(ncols =2)

mat.plotContactMatrix(school_partition, schools, title = "Contact in schoolplaces by household income")#, ax = ax[0])
plt.show()
mat.plotContactMatrix(employed_partition, workplaces, title = "Contact in workplaces by household income")
plt.show()
mat.plotContactMatrix(full_partition, households, title = "contact matrix for households")
plt.show()


model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)
#get and plot a distribution of incomes for each household
#model.infectPopulace(init_infection_rate)
#model.simulate(gamma,tau)
#rat.plotBars(model, full_partition, ip)
#plt.show()
#plot a distribution of incomes

#hh_incomes = [hh.income for hh in households.values()]
#getIncome = lambda person: households[person.sp_hh_id].income
#personal_incomes = [households[pers['sp_hh_id']].income for pers in model.populace]
#sns.distplot(hh_incomes, kde = True, axlabel = 'income',label = 'household_incomes')
#sns.distplot(personal_incomes, kde = True, axlabel = 'person_incomes')
#plt.show()