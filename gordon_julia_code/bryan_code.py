# Written by Bryan Azbill, 2020-06-01
import time
import EoN
import networkx as nx
from networkx import random_graphs # GE
import random
import numpy as np
import matplotlib.pyplot as plt
import math

schoolGroupSize = 20 # GE
epidemicSims = 5
houseHolds = 5000
houseHoldSize = 20
population = houseHoldSize*houseHolds
classSize = 20
workGroupSize = 10
employmentRate = 0.9
recoveryRate = 1 #
globalInfectionRate = 1

globalInfectionRate = 1
houseInfectivity = .1
workClasses = ['default','unemployed','school']
workInfectivity = .05

ageGroups = [[0,5], [5,18], [18,65], [65,90]]
ageGroupWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019

start = time.time()
tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1

## enumerate population
citizens = list(range(population))# a number for each citizen (node)

## assign households
houseNumbers = [i%houseHolds for i in range(population)]
#stochastic age assignment
ageGroups = [random.choices(ageGroups, weights = ageGroupWeights)[0] for i in range(population)]

#assign work or school place with stochastic unemployment
def genWorkType(ageGroup):    #school and employment
    if(ageGroup == [0,5]):#preschool
        workType = 'none'
    elif(ageGroup == [5,18]):#TODO split into grade/middle/high
        workType = 'school'
    else:
        if(random.uniform(0,1)<employmentRate):
            workType = 'default'
        else:
            workType = 'none'
    return workType
workTypes = [genWorkType(age) for age in ageGroups]

citizenWorkTypes = list(zip(workTypes,citizens))
#TODO put these in loops and arrays
working = list(list(zip(*(filter(lambda x: x[0] == 'default',citizenWorkTypes))))[1])
students = list(list(zip(*(filter(lambda x: x[0] == 'school',citizenWorkTypes))))[1])
unemployed = list(list(zip(*(filter(lambda x: x[0] == 'none',citizenWorkTypes))))[1])
random.shuffle(working)
random.shuffle(students)

classCount = int(math.ceil(len(students) / schoolGroupSize))
workgroupCount = int(math.ceil(len(students) / schoolGroupSize))
environmentCount  = classCount+workgroupCount
schoolAssignments = list(zip(students,[i%classCount for i in range(len(students))]))
workAssignments   = list(zip(working,[i%workgroupCount+classCount for i in range(len(working))]))
unassigned        = list(zip(unemployed,[None for i in range(len(unemployed))]))

workAssignments.extend(unassigned)
schoolAssignments.extend(workAssignments)
assignments = list(zip(*sorted(schoolAssignments)))[1]

sims = np.array([citizens,houseNumbers,workTypes,assignments])
graph = nx.Graph()
graph.add_nodes_from(list(range(population)))

#distribute workers into different environments

#function to create homogeneous group
def groupCitizens(graph, citizens, weight):
    groupSize = len(citizens)
    for i in range(groupSize):
        for j in range(i):
            graph.add_edge(citizens[j],citizens[i],transmission_weight = weight)


#link population in the same households
citizenHouses = list(zip(citizens,houseNumbers))
for i in range(houseHolds):
    house = list(zip(*list(filter(lambda x: (x[1]==i),citizenHouses))))[0]
    groupCitizens(graph, house, houseInfectivity)

#link population in the same work environmen
assignmentGroups = list(zip(citizens, assignments))
for i in range(environmentCount):
    environmentGroup = list(zip(*list(filter(lambda x:(x[1]==i),assignmentGroups))))[0]
    groupCitizens(graph, environmentGroup, workInfectivity)


end = time.time()
print(end - start)
start = time.time()
#for i in range(epidemicSims):
node_investigation = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.01, transmission_weight ='transmission_weight',return_full_data = True)
plt.plot(node_investigation.summary(students)[1]['I'],label = "infected students")
plt.plot(node_investigation.summary(working)[1]['I'],label = "infected workers")
plt.plot(node_investigation.summary(unemployed)[1]['I'],label = "infected unemployed")
plt.legend()
plt.show()
#plt.plot(t,R)
#plt.plot(t,I)
#plt.plot(t,S)
#plt.show()
end = time.time()
print(end - start )
