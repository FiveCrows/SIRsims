
# Written by Bryan Azbill, 2020-06-01
import EoN
import networkx as nx
from networkx import random_graphs # GE
import random
import numpy as np
import matplotlib.pyplot as plt
import math

epidemicSims = 5
houseHolds = 500
houseHoldSize = 20
population = houseHoldSize*houseHolds
classSize = 20
workGroupSize = 10
employmentRate = 0.9
recoveryRate = 1 #
globalInfectionRate = 1
houseInfectivity = .1
workClasses = ['default','unemployed','school']
workInfectivity = .05


ageGroups = []
tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1
G = nx.Graph()

#
## generate population
population = int(2.0 * population) # GE
citizens = list(range(population))# a number for each citizen (node)

## assign households
houseNumbers = [i%houseHolds for i in range(population)]

#stochastic age assignment
ageChoices = [':5', '5:18', '18:65', '65:']
ageWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
ages = [random.choices(ageChoices, weights = ageWeights)[0] for i in range(population)]

#assign work or school place with stochastic unemployment
def genWorkType(age):    #school and employment
    if(age == ':5'):#preschool
        workType = 'none'
    elif(age == '5:18'):#TODO split into grade/middle/high
        workType = 'school'
    else:
        if(random.uniform(0,1)<employmentRate):
            workType = 'default'
        else:
            workType = 'none'#workGroupSize = 10
    return workType

workTypes = [genWorkType(age) for age in ages]
citizenWorkTypes = list(zip(workTypes,citizens))
#TODO put these in loops and arrays
working = list(list(zip(*(filter(lambda x: x[0] == 'default',citizenWorkTypes))))[1])
students = list(list(zip(*(filter(lambda x: x[0] == 'school',citizenWorkTypes))))[1])
unemployed = list(list(zip(*(filter(lambda x: x[0] == 'none',citizenWorkTypes))))[1])
random.shuffle(working)
random.shuffle(students)

classCount = int(math.ceil(len(students)/classSize))
workgroupCount = int(math.ceil(len(students)/classSize))
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

# Number of nodes and edges in this graph
graph = random_graphs.random_regular_graph(10, 20000)
print("nb nodes: ", graph.number_of_nodes())
print("nb edges: ", graph.number_of_edges())

#nx.draw(graph)
#plt.show()

# Execute 5 runs
print("Execute 5 runs")
for i in range(epidemicSims):
    print("run %d" % i)
    #t,S,I,R = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.01, transmission_weight ='transmission_weight')
    t,S,I,R = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.01) #, transmission_weight ='transmission_weight')
    plt.plot(t,R)
    plt.plot(t,I)
    plt.plot(t,S)
plt.xlabel("time")
plt.ylabel("citizens")
plt.show()
