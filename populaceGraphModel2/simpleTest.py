from modelingToolkit import *
import json

# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(prevention_adoptions, prevention_efficacies)

#initialize NetBuilder and build networks
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies)
model.buildNetworks(netBuilder)

#simulate
model.simulate(gamma,tau)
model.plotSIR()