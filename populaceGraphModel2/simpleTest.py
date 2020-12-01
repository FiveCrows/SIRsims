from modelingToolkit import *
import json

# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = True)

#initialize NetBuilder and build networks
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies)
model.networkEnvs(netBuilder)
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

model.infectPopulace(0.01)

#simulate
model.simulate(gamma,tau)
model.plotSIR()
#