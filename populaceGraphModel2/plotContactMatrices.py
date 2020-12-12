#this function generates a histogram for household sizes
from modelingToolkit import *
from resultAnalysisToolkit import *
import numpy as np
import json

# Whether or not to save output files  <<<<<<<<<<<<<< Set to save directory
defaultParams  = (json.load(open('defaultParameters')))
globals().update(defaultParams)
#construct partitioner


# denotes the fraction of people using masks

#initialize populaceGraph

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
#doesn't really matter, just to put in place

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

slim = False

model = PopulaceGraph()
model.networkEnvs(NetBuilder())
households = model.listEnvByType('household')
schools = model.listEnvByType('school')
workplaces = model.listEnvByType('workplace')
#fig, axs = plt.subplots(1,3, True, True)

partitioner = Partitioner.agePartitionerA()
plotContactMatrix(model, partitioner, households, " all households", ax = plt)
plt.show()
plotContactMatrix(model, partitioner, schools, " all schools", ax = plt)
plt.show()
plotContactMatrix(model, partitioner, workplaces, " all workplaces", ax = plt)
plt.show()