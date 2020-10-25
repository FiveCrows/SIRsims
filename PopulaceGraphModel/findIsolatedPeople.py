"""
This script is written to check who has no edges in the graph
"""

from ModelToolkit import *
import copy


#These values scale the weight that goes onto edges by the environment type involved
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1} # base case
#As None, the degrees of the environment are implicit in contact matrices
env_degrees = {'workplace': None, 'school': None}
#the prevention measures in the workplaces
workplace_preventions = {'masking': 0.0, 'distancing': 0}
#the prevention measures in the schools
school_preventions = {'masking':0.0, 'distancing': 0}
#the prevention measures in the household
household_preventions = {'masking':0.0, 'distancing':0}
#combine all preventions into one var to easily pass during reweight and build
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
#these values specify how much of a reduction in weight occurs when people are masked, or distancing
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values
#this object holds rules and variables for choosing weights
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
#gamma is the recovery rate, and the inverse of expected recovery time
gamma = 0.1
#tau is the transmission rate
tau = 0.08

#the partioner is needed to put the members of each environment into a partition,
#currently, it is setup to match the partition that is implicit to the loaded contact matrices
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner('age', enumerator, names)

#----------- BEGIN SIMULATIONS ------------------------------------

#init, build simulate
model = PopulaceGraph( partition, slim = False)
model.build(trans_weighter, preventions, env_degrees)
model.differentiateMasks([0.25,0.5,0.25])
print(model.populace.keys()-model.graph.nodes())
