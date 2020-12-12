from modelingToolkit import *
import json
import seaborn as sns
# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = False)
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

#get incomes for each household 
households = model.listEnvByType('households')
incomes = list(set([model.environments[i].__dict__['hh_income'] for i in households]))
sns.distplot(income, kde = true, axlabel = 'income')
plt.show()
