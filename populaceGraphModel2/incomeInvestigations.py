##################################################################
#This script is here to reveal information about the incomes of the population
##################################################################

from modelingToolkit import *
from resultAnalysisToolkit import *
import json
import seaborn as sns

# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = True)
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

#get and plot a distribution of incomes for each household 
envs = model.environments
households = filterEnvByType(envs,'household')
#plot a distribution of incomes

hh_incomes = [hh.hh_income for hh in households.values()]

#personal_incomes = [households[pers['sp_hh_id']].hh_income for pers in model.populace]
sns.distplot(hh_incomes, kde = True, axlabel = 'income',label = 'household_incomes')
sns.distplot(personal_incomes, kde = True, axlabel = 'person_incomes')
plt.show()

#construct an enumerator such that each partition gets an even number of members for income
#number of partitions to have for incomes
nbins = 10
personal_incomes.sort()
boundaries.append(personal_incomes[-1])
#enumerator = [i: ]
#davis scrub python 
binwidth = len(person_incomes)//nbins
enumerator = [personal_incomes[i]: i//binwidth for i in range(len(personal_incomes))]


    
    
    
        