#this function generates a histogram for household sizes
from ge_modelingToolkit2 import *
import numpy as np

gamma = 0.2# The expected time to recovery will be 1/gamma (days)
tau = 0.2#  The expected time to transmission by an ge will be 1/(weight*Tau) (days)

# Whether or not to save output files  <<<<<<<<<<<<<< Set to save directory
save_output = False
save_output = True
print("save_output= ", save_output)

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)

# denotes the fraction of people using masks
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}
#initialize populaceGraph
slim = False
print("Running with slim= %d" % slim)

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
#doesn't really matter, just to put in place
env_type_scalars = {"household": 1, "school": 1.0, "workplace": 1.0}
prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

slim = False
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies)
model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim)
model.buildNetworks(netBuilder)
households = model.listEnvByType('household')
schools = model.listEnvByType('school')
workplaces = model.listEnvByType('workplace')
model.plotContactMatrix(partitioner, households, " all households")
model.plotContactMatrix(partitioner, schools, " all schools")
model.plotContactMatrix(partitioner, workplaces, " all workplaces")

print("ok")
