from modelingToolkit import *
import json

# load default params for model
defaultParams  = (json.load(open('defaultParameters')))
globals().update(defaultParams)
#initialize model
model = PopulaceGraph(slim = True)

#initialize NetBuilder and build networks
netBuilder = NetBuilder()
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

model.infectPopulace(initial_infection_rate)

#simulate
#model.simulate(gamma,tau)
#model.plotSIR()
#