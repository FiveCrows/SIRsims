import numpy as np
import scipy as sci
import networkx as nx
import pandas as pd
import random
import pickle
import itertools
import matplotlib.pyplot as plt
import EoN
import time
import math


epidemicSims = 10
houseHolds = 10
houseHoldSize = 500
people = houseHoldSize * houseHolds
schoolGroupSize = 20
workGroupSize = 10
employmentRate = 0.9
recoveryRate = 1
globalInfectionRate = 1
homeInfectivity = 1
schoolInfectivity = 0.5
workInfectivity = 0.5
tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1

#ageGroups = [[0,5], [5,8], [18,65], [65,90]]
#ageGroupWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
attributes = {"age": ['[0,5]', '[5,18]', '[18,65]', '[65,90]'], "gender": ['M', 'F']}
attribute_p = {"age": [0.05, 0.119, 0.731, 0.1], "gender": [0.5,0.5]}
duties = [None, 'school','work']

#incomeGroups =
#incomeGroupWeights =


class Person():
    attributes = {}
# a function which returns a list of tuples randomly assigning nodes to groups of size n
def nGroupAssign(members, groupSize):
    length = len(members)
    random.shuffle(members)
    pos = 0
    groupNumber = 0
    dict = {}
    while True:
        if(pos+groupSize>length):
            dict[groupNumber] = (itertools.islice(members, pos, pos + groupSize))
            break
        dict[groupNumber] = list(itertools.islice(members, pos, pos + groupSize))
        groupNumber = groupNumber + 1
        pos = pos+groupSize
    return dict


# a function which returns a list of tuples randomly assigning nodes to groups of size probability n
def p_nGroupAssign(memberIndices, p_n):
    length = len(memberIndices)
    random.shuffle(memberIndices)
    pos = 0
    groupNumber = 0
    dict = {}
    while True:
        groupSize = random.choices(range(len(p_n)), weights=p_n)[0]+1
        if(pos+groupSize>length):
            dict[groupNumber] = (itertools.islice(memberIndices, pos, pos + groupSize))
            break
        dict[groupNumber] = list(itertools.islice(memberIndices, pos, pos + groupSize))
        groupNumber = groupNumber + 1
        pos = pos+groupSize
    return dict

def p_attributeAssign(memberIndices, attributes, probabilities):
    random.shuffle(memberIndices)
    dict = {attribute: [] for attribute in attributes}
    for index in memberIndices:
        assignment = random.choices(attributes, weights = probabilities)[0]
        dict[assignment].append(index)
    return dict



#for loading people objects from file
def loadPickledPop(filename):
    with open(filename,'rb') as file:
        x = pickle.load(file)
    #return represented by dict of dicts
    populace = ({key: (vars(x[key])) for key in x})#.transpose()
    #csv = pd.DataFrame.from_dict(populace)
    #csv.to_csv("./datasets/synthPopulace.csv")
    return populace

# assign people to households


def genPop(people, attributeClasses, attributeClass_p):
    population = {i: {} for i in range(people)}
    for attributeClass in attributeClasses:
        assignments = p_attributeAssign(list(range(people)), attributeClasses[attributeClass],attributeClass_p[attributeClass])
        for  key in assignments:
            for i in assignments[key]:
                population[i][attributeClass] = key
    
    return population


#takes a dict of dicts to represent populace and returns a list of dicts of lists to represent groups of people with the same
#attributes
def sortPopulace(populace, categories):
    groups = {category: {} for category in categories}
    for person in populace:
        for category in categories:
            try:
                groups[category][populace[person][category]].append(person)
            except:
                groups[category][populace[person][category]] = [person]
    return groups


#connect list of groups with weight
#TODO update to use a weight calculating function
def clusterDenseGroups(graph, groups, weight):
    for key in groups.keys():
        if key !=None:
            memberCount = len(groups[key])
            memberWeightScalar = np.sqrt(memberCount)
            for i in range(memberCount):
                for j in range(i):
                    graph.add_edge(groups[key][i],groups[key][j], transmission_weight = weight/memberWeightScalar)






def clusterByDegree_p(graph, groups, weight,degree_p):
    #some random edges may be duplicates, best for large groups
    connectorList = []

    for key in groups.keys():
        if key !=None:
            memberCount = len(groups[key])
            connectorList = []
            for i in range(memberCount):
                nodeDegree = random.choices(range(len(degree_p)), weights = degree_p)
                connectorList.extend([i]*nodeDegree[0])
            random.shuffle(connectorList)

            i = 0
            while i < len(connectorList)-1:
                graph.add_edge(groups[key][connectorList[i]],groups[key][connectorList[i+1]],transmission_weight = weight)
                i = i+2




def clusterWith_gnp_random(graph,groups,weight,avgDegree):
    for key in groups.keys():
        if key !=None:
            memberCount = len(groups[key])
            edgeProb = (memberCount*avgDegree)/(memberCount*(memberCount-1))
            graph2 = nx.fast_gnp_random_graph(memberCount,edgeProb)
            graph.add_edges_from(graph2.edges())


#clusters groups into strogatz small-worlds networks
def strogatzDemCatz(graph, groups, weight, local_k, rewire_p):
    if(local_k%2!=0):
        print("Error: local_k must be even")

    for key in groups:
        if key!=None:
            memberCount = len(groups[key])
            if local_k >= memberCount:
                print("warning: not enough members in group for {}".format(local_k) + "local connections in strogatz net")
                local_k = memberCount-1

            group = groups[key]
            #unfinished for different implementation to not leave any chance of randomly selecting the same edge twice
            #rewireCount = np.random.binomial(memberCount, rewire_p)
            #rewireList = np.choices(group, rewireCount)*2

            for i in range(memberCount):
                nodeA = group[i]
                for j in range(-local_k//2, local_k//2+1):
                    if j == 0:
                        continue
                    rewireRoll = random.uniform(0,1)

                    if rewireRoll<rewire_p:
                        nodeB = group[(i + random.choice(range(memberCount - 1))) % memberCount]
                    else:
                        nodeB = group[(i+j)%memberCount]
                    graph.add_edge(nodeA, nodeB, transmission_weight=weight)

#WIP
def clusterGroupsByPA(graph, groups):
    for key in groups.keys():
        memberCount = len(groups[key])


def showGroupComparison(sim, category, groupTags, popsByCategory, title = None):
        for groupTag in groupTags:
            group = popsByCategory[category][groupTag]
            plt.plot(sim.summary(group)[1]['I']/len(group),label = "{}: {}".format(category,groupTag))
        plt.legend()
        plt.title(title)
        plt.ylabel("percent infected")
        plt.xlabel("time steps")
        plt.show()


#def displayGroupNetwork():
#def sumNodeWeights(graph, node):



#def mergeSubClusterGraph(graph,subgraph, nodeMap):
#def sortAttributes(people,attributeClasses):
#populace = genPop(people, attributes, attribute_p)

print("loading and sorting populations")
start = time.time()
populace = loadPickledPop("people_list_serialized.pkl")
popsByCategory = sortPopulace(populace, ['sp_hh_id', 'work_id', 'school_id', 'race'])
graph1 = nx.Graph()
graph2 = nx.Graph()
stop = time.time()
print("finished in {} seconds".format(stop - start))

print("building populace into graphs")
start = time.time()

#clusterDenseGroups(graph1, popsByCategory['sp_hh_id'],homeInfectivity)
clusterDenseGroups(graph2, popsByCategory['sp_hh_id'],homeInfectivity)
#clusterByDegree_p(graph1,popsByCategory['work_id'], 1, [0,0,0.2,0.3,0.5])
#clusterByDegree_p(graph1,popsByCategory['school_id'], 1, [0,0,0.2,0.3,0.5])
strogatzDemCatz(graph2, popsByCategory['work_id'], workInfectivity, 4, 0.1)
strogatzDemCatz(graph2, popsByCategory['school_id'],schoolInfectivity, 4,0.1)
stop = time.time()
print("finished in {} seconds".format(stop - start))

start = time.time()
print("running event-based simulation")
node_investigation1 = EoN.fast_SIR(graph1, globalInfectionRate, recoveryRate, rho = 0.0001, transmission_weight ='transmission_weight',return_full_data = True)
node_investigation2 = EoN.fast_SIR(graph2, globalInfectionRate, recoveryRate, rho = 0.0001, transmission_weight ='transmission_weight',return_full_data = True)
stop = time.time()
print("finished in {} seconds".format(stop - start))

#showGroupComparison(node_investigation1, 'race', [1,2], popsByCategory, "built with random nets")
showGroupComparison(node_investigation2, 'race', [1,2], popsByCategory, "built with strogatz nets")

#node_investigation.animate(popsByCategory['school_id'][450143554])

#if not nx.is_connected(graph):
#    print("warning: graph is not connected, there are {} components".format(nx.number_connected_components(graph.subgraph(popsByCategory['work_id'][505001334]))))

#node_investigation.animate()



#plt.plot(node_investigation.summary(popsByCategory['race'][2])[1]['I']/racePops[1],label = "infected students")
#plt.plot(node_investigation.summary(popsByCategory['race'][3])[1]['I']/racePops[2],label = "infected students")
#plt.plot(node_investigation.summary(graph,label = "infected students")



