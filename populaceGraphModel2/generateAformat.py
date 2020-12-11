###############################################################################
#this script is written to export a network for testing with other scripts
##############################################################################


from modelingToolkit import *
from resultExportsToolkit import aFormatGraph
import json
#this short script generates the example exported graph, with nodes and edges 
# load default params for model
globals().update(json.load(open('defaultParameters')))

#initialize model
model = PopulaceGraph(slim = False)

#initialize NetBuilder and build networks
model.networkEnvs(NetBuilder())
model.weightNetwork(env_type_scalars, prevention_adoptions, prevention_efficacies)

aFormatGraph(model, 'fullAformatExample')






