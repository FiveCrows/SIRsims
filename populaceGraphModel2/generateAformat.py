from modelingToolkit import *
from resultAnalysisToolkit import aFormatGraph
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






