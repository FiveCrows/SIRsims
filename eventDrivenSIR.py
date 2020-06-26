# Written by Bryan Azbill, 2020-06-01
from os import mkdir

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
from datetime import datetime
import math


#for generating population
houseHolds = 10
houseHoldSize = 500
people = houseHoldSize * houseHolds
schoolGroupSize = 20
workGroupSize = 10
employmentRate = 0.9

#for generating graph
graphGenParams = 1
globalInfectionRate = 1
homeInfectivity = 1
schoolInfectivity = 0.5
workInfectivity = 0.5
workAvgDegree = 10
schoolAvgDeree = 10
timeCode = True
tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1


#for recording results



#ageGroups = [[0,5], [5,8], [18,65], [65,90]]
#ageGroupWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
attributes = {"age": ['[0,5]', '[5,18]', '[18,65]', '[65,90]'], "gender": ['M', 'F']}
attribute_p = {"age": [0.05, 0.119, 0.731, 0.1], "gender": [0.5,0.5]}
duties = [None, 'school','work']

#incomeGroups =
#incomeGroupWeights =
class Record:
    def __init__(self):
        self.log = ""
        self.stamp = datetime.now().strftime("%m_%d_%H_%M")
    def print(self, string):
        print(string)
        self.log+=('\n')
        self.log+=(string)

    def dump(self):
        mkdir("./simResults/{}".format(self.stamp))
        log_txt = open("./simResults/{}/log.txt".format(self.stamp),"w+")
        log_txt.write(self.log)

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
    csv = pd.DataFrame.from_dict(populace).transpose()
    csv.to_csv("./datasets/synthPopulaceReformat.csv")
    return populace


#def bubblePlot():
# assign people to households


#Work in progress
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
def clusterDenseGroup(graph, group, memberCount, weight, params = None):
    subGraph = nx.complete_graph(memberCount)
    relabel = dict(zip(range(memberCount), group))
    subGraph = nx.relabel.relabel_nodes(subGraph,relabel)
    graph.add_edges_from(subGraph.edges(), transmission_weight=weight)
#supports dense (no params), degree_p (params is prob list), random (param is avg_degree), strogatz, (degree and rewire_p), and preferential attachment

def clusterRandom(graph,group, memberCount, weight, params):
    avg_degree = params
    if avg_degree >= memberCount:
        clusterDense(graph, group, memberCount, weight, params)
        return
    edgeProb = 2*avg_degree / (memberCount - 1)
    subGraph = nx.fast_gnp_random_graph(memberCount, edgeProb)
    relabel = dict(zip(range(memberCount), group))
    nx.relabel.relabel_nodes(subGraph, relabel)
    graph.add_edges_from(subGraph.edges(), transmission_weight=weight)

def clusterDense(graph, group, memberCount, weight, params):
    #memberWeightScalar = np.sqrt(memberCount)
    for i in range(memberCount):
        for j in range(i):
            graph.add_edge(group[i], group[j], transmission_weight=weight) #/ memberWeightScalar)

def clusterDegree_p(graph,group, memberCount, weight, params):
    degree_p = params
    connectorList = []
    for i in range(memberCount):
        nodeDegree = random.choices(range(len(degree_p)), weights=degree_p)
        connectorList.extend([i] * nodeDegree[0])
    random.shuffle(connectorList)
    # this method DOES leave the chance adding duplicate edges
    i = 0
    while i < len(connectorList) - 1:
        graph.add_edge(group[connectorList[i]], group[connectorList[i + 1]],
                       transmission_weight=weight)
        i = i + 2


#def clusterPartition(graph, group, memberCount, weight, params):
def clusterStrogatz(graph,group, memberCount, weight, params):
    group.sort()
    local_k = params[0]
    rewire_p = params[1]
    if (local_k % 2 != 0):
        record.print("Error: local_k must be even")
    if local_k >= memberCount:
        clusterDense(graph, group, memberCount, weight, params)
        return

    for i in range(memberCount):
        nodeA = group[i]
        for j in range(1, local_k // 2+1):
            if j == 0:
                continue
            rewireRoll = random.uniform(0, 1)
            if rewireRoll < rewire_p:
                nodeB = group[(i + random.choice(range(memberCount - 1))) % memberCount]

            else:
                nodeB = group[(i + j) % memberCount]
            graph.add_edge(nodeA, nodeB, transmission_weight=weight)


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


def clusterGroupsByPA(graph, groups):
    for key in groups.keys():
        memberCount = len(groups[key])


def clusterGroups(graph, classifier, weight, clusterAlg, params = None):
    record.print('\n')
    record.print("clustering {} groups with the {} algorithm".format(classifier, clusterAlg.__name__))
    start = time.time()

    groups = popsByCategory[classifier]
    initial_weights = graph.size()
    for key in groups.keys():
        if key == None:
            continue
        group = groups[key]
        clusterAlg(graph, group, len(group), weight, params)
    weights_added = graph.size() - initial_weights
    stop = time.time()
    record.print("{} weights of size {} added for {} work environments in {} seconds".format(weights_added, weight,len(popsByCategory[classifier].keys()), stop-start))



def showGroupComparison(sim, category, groupTags, popsByCategory, node_investigation):
        for groupTag in groupTags:
            group = popsByCategory[category][groupTag]
            plt.plot(node_investigation.summary(group)[1]['I']/len(group),label = "{}: {}".format(category,groupTag))
        plt.legend()
        plt.ylabel("percent infected")
        plt.xlabel("time steps")
        plt.show()


#def summarizeGroup(group):
#def mergeSubClusterGraph(graph,subgraph, nodeMap):
#def sortAttributes(people,attributeClasses):
#populace = genPop(people, attributes, attribute_p)


def simulateGraph(clusteringAlg, params, full_data = False):
    record.print('\n')
    record.print("building populace into graphs with the  {} clustering algorithm".format(clusteringAlg.__name__))
    start = time.time()

    graph = nx.Graph()
    clusterGroups(graph, 'sp_hh_id', homeInfectivity, clusterDenseGroup)
    record.print("{} weights of size {} have been added for {} homes".format(graph.size(), 1, len(popsByCategory['sp_hh_id'].keys())))


    clusterGroups(graph, 'work_id', workInfectivity, clusteringAlg, params)
    clusterGroups(graph, 'school_id', workInfectivity, clusteringAlg, params)

    stop = time.time()
    record.print("The final graph finished in {} seconds. with properties:".format((stop - start)))
    record.print("{edges: {}, nodes: }".format(graph.size()))
    return(graph)
    record.print("running event-based simulation")

    if full_data:
        simResult = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho=0.0001,transmission_weight='transmission_weight', return_full_data=True)
    else:
        simResult = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho=0.0001,transmission_weight='transmission_weight', return_full_data=False)
    stop = time.time()
    record.print("finished in {} seconds".format(stop - start))
    return simResult


record = Record()
record.print( "loading and sorting synthetic populations")
start = time.time()
populace = loadPickledPop("people_list_serialized.pkl")
popsByCategory = sortPopulace(populace, ['sp_hh_id', 'work_id', 'school_id', 'race'])
stop = time.time()
record.print("finished in {} seconds".format(stop - start))






t,S,I,R, = simulateGraph(clusterRandom,workAvgDegree)
plt.plot(t,I,label = 'random')

t,S,I,R =simulateGraph(clusterStrogatz,workAvgDegree)
plt.plot(t,I,label = 'strogatz')


#node_investigation = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.0001, transmission_weight ='transmission_weight',return_full_data = True)
#showGroupComparison(node_investigation, 'race', [1,2], popsByCategory)
#node_investigation.animate(popsByCategory['school_id'][450143554])

#if not nx.is_connected(graph):
#    print("warning: graph is not connected, there are {} components".format(nx.number_connected_components(graph.subgraph(popsByCategory['work_id'][505001334]))))
#node_investigation.animate()


plt.xlabel("time")
plt.ylabel("infected count")
plt.legend()
record.dump()
plt.savefig("./simResults/{}/plot".format(record.stamp))
plt.show()
#plt.plot(node_investigation.summary(popsByCategory['race'][3])[1]['I']/racePops[2],label = "infected students")
#plt.plot(node_investigation.summary(graph,label = "infected students")



