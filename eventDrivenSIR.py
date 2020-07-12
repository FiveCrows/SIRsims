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
import multiprocessing

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
workAvgDegree = 20
schoolAvgDeree = 20
timeCode = True
tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1

#for simulating graph
recovery_rate = 0.1
globalInfectionRate = 0.08

#for recording results
#ageGroups = [[0,5], [5,8], [18,65], [65,90]]
#ageGroupWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
#attributes = {"age": ['[0,5]', '[5,18]', '[18,65]', '[65,90]'], "gender": ['M', 'F']}
#attribute_p = {"age": [0.05, 0.119, 0.731, 0.1], "gender": [0.5,0.5]}
#duties = [None, 'school', 'work']

#incomeGroups =
#incomeGroupWeights =

class Person():
    attributes = {}
# a function which returns a list of tuples randomly assigning nodes to groups of size n



class Record:
    def __init__(self):
        self.log = ""
        self.comments = ""
        self.stamp = datetime.now().strftime("%m_%d_%H_%M")
        self.graph_stats = {}
        self.last_runs_percent_uninfected = 1
    def print(self, string):
        print(string)
        self.log+=('\n')
        self.log+=(string)

    def getComment(self):
        self.comments += input("Enter comment")

    def printGraphStats(self, graph, statAlgs):
        if not nx.is_connected(graph):
            self.print("graph is not connected. There are {} components".format(nx.number_connected_components(graph)))
            max_subgraph = graph.subgraph(max(nx.connected_components(graph)))
            self.print("{} of nodes lie within the maximal subgraph".format(max_subgraph.number_of_nodes()/graph.number_of_nodes()))
        else:
            max_subgraph = graph
        graphStats = {}
        for statAlg in statAlgs:
            graphStats[statAlg.__name__] = statAlg(max_subgraph)
        self.print(str(graphStats))

    def dump(self):
        mkdir("./simResults/{}".format(self.stamp))
        log_txt = open("./simResults/{}/log.txt".format(self.stamp),"w+")
        log_txt.write(self.log)
        if self.comments != "":
            comment_txt = open("./simResults/{}/comments.txt".format(self.stamp),"w+")
            comment_txt.write(self.comments)

#WIP
class Environment:
    def __init__(self):
        print("WIP")

    #def plot_homes(self):

#WIP
#TODO refonfigure masking
class TransmissionWeighter:
    def __init__(self, loc_scalars, mask_scalar):
        self.name = 'sole'
        self.global_weight = 1
        self.mask_scalar = mask_scalar
        self.loc_scalars = loc_scalars

        #self.age_scalars = age_scalars

    def getWeight(self, personA, personB, location, masking):
        weight = self.global_weight
        try:
            weight = weight*self.loc_scalars[location]
        except:
            print("locale type not identified")

        if (masking != None):
            if random.random()<masking:
                weight = weight*self.mask_scalar
        if masking != None:
            if random.random()<masking:
                weight = weight*self.mask_scalar
        return weight


    #WIP
    def reweight(graph, groups):
        for edge in graph.edges():
            print("stub")
            #graph[edge[0],]

    #WIP
    def record(self, record):
        record.print("A transmission probability weighter has been configured with:".format(mask_scalar))
        record.print(str(self.__dict__))






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
def loadPickles(filename):
    with open(filename,'rb') as file:
        x = pickle.load(file)
    #return represented by dict of dicts
    populace = ({key: (vars(x[key])) for key in x})#.transpose()
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

def getWeight(person_A, person_B, locale):
    return 1

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

def clusterRandom(graph, group, location,  member_count, weighter, masking, params):
    avg_degree = params
    if avg_degree >= member_count:
        clusterDense(graph, group, member_count, weighter, params)
        return
    edgeProb = 2 * avg_degree / (member_count - 1)

    if member_count < 100:  # otherwise this alg is too slow
        total_edges = avg_degree * member_count
        pos_edges = itertools.combinations(group,2)
        for edge in pos_edges:
            if random.random()<edgeProb:
                graph.add_edge(edge[0],edge[1], transmission_weight = weighter.getWeight(edge[0],edge[1], location, masking))

    else:
        for i in range(member_count-1):
            nodeA = group[i]
            for j in range(i+1,member_count):
                if random.random()<edgeProb:
                    nodeB = group[j]
                    graph.add_edge(nodeA,nodeB, transmission_weight = weighter.getWeight(nodeA,nodeB,location))

#WIP
def clusterPartitions(graph, group, location, member_count, weighter, masking, params):
    partition_size = params[0]
    mixing_rate = params[1]
    if partition_size>member_count:
        clusterDense(graph, group, member_count, weighter, masking, params)
        return
    #groups = nGroupAssign()


def clusterDense(graph, group, location, member_count, weighter, masking, params):
    #memberWeightScalar = np.sqrt(memberCount)
    for i in range(member_count):
        for j in range(i):
            graph.add_edge(group[i], group[j], transmission_weight=weighter.getWeight(group[i],group[j], location, masking)) #/ memberWeightScalar)


def clusterDegree_p(graph,group, location, memberCount, weighter, masking, params):
    degree_p = params
    connectorList = []
    for i in range(memberCount):
        nodeDegree = random.choices(range(len(degree_p)), weights=degree_p)
        connectorList.extend([i] * nodeDegree[0])
    random.shuffle(connectorList)
    # this method DOES leave the chance adding duplicate edges
    i = 0
    while i < len(connectorList) - 1:
        nodeA = group[connectorList[i]]
        nodeB = group[connectorList[i + 1]]
        graph.add_edge(nodeA, nodeB, transmission_weight = weighter.getWeight(nodeA,nodeB, location, masking))
        i = i + 2


def clusterStrogatz(graph, group, location, memberCount, weighter, masking, params):
    group.sort()
    local_k = params[0]
    rewire_p = params[1]
    if (local_k % 2 != 0):
        record.print("Error: local_k must be even")
    if local_k >= memberCount:
        clusterDense(graph, group, location, memberCount, weighter, masking, params)
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
            graph.add_edge(nodeA, nodeB, transmission_weight=weighter.getWeight(nodeA,nodeB, location, masking))


def clusterByDegree_p(graph, groups, weighter, masking, degree_p):
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
                nodeA = groups[key][connectorList[i]]
                nodeB = groups[key][connectorList[i+1]]
                graph.add_edge(nodeA,nodeB,transmission_weight = weighter.getWeight(nodeA,nodeB))
                i = i+2


def clusterGroupsByPA(graph, groups):
    for key in groups.keys():
        memberCount = len(groups[key])


def clusterGroups(graph, classifier, transmissionWeighter, clusterAlg, masking, params = None):
    record.print("clustering {} groups with the {} algorithm".format(classifier, clusterAlg.__name__))
    start = time.time()
    # # stats = {"classifier": }
    groups = popsByCategory[classifier]
    group_count = len(groups)

    initial_weights = graph.size()
    for key in groups.keys():
        if key == None:
            continue
        group = groups[key]
        clusterAlg(graph, group, classifier, len(group), transmissionWeighter, masking, params)

    weights_added = graph.size() - initial_weights
    stop = time.time()
    record.print("{} weights added for {} environments in {} seconds".format(weights_added,len(popsByCategory[classifier].keys()), stop-start))

#def clusterBlendedGroups(graph, groups, contact_matrix)

def showGroupComparison(sim, category, groupTags, popsByCategory, node_investigation, record):
        record.print("plotting an infection rate comparison of groups in category of {}".format(category))
        for groupTag in groupTags:
            group = popsByCategory[category][groupTag]
            record.print_and_record("adding {} group to plot, it has {} mem".format(groupTag))
            plt.plot(node_investigation.summary(group)[1]['I']/len(group),label = "{}: {}".format(category,groupTag))
        plt.legend()
        plt.ylabel("percent infected")
        plt.xlabel("time steps")
        plt.show()



def simulateGraph(clusteringAlg, simAlg, transmissionWeighter, params = None, full_data = False, exemption = None, masking = {'schools': None, 'workplaces': None}):
    record.print('\n')
    record.print("building populace into graphs with the {} clustering algorithm".format(clusteringAlg.__name__))
    start = time.time()

    graph = nx.Graph()
    clusterGroups(graph, 'sp_hh_id', transmissionWeighter, clusterDense, None)

    if exemption != 'workplaces':
        clusterGroups(graph, 'work_id', transmissionWeighter, clusteringAlg, masking['workplaces'], params)
    if exemption != 'schools':
        clusterGroups(graph, 'school_id', transmissionWeighter, clusteringAlg, masking['schools'], params)

    stop_a = time.time()
    record.print("Graph completed in {} seconds.".format((stop_a - start)))
    #record.printGraphStats(graph, [nx.average_clustering])
    # record.print("{edges: {}, nodes: }".format(graph.size()))
    record.print("running simulation with the {} algorithm".format(simAlg.__name__))

    if full_data:
        simResult = simAlg(graph, globalInfectionRate, recovery_rate, rho=0.0001,transmission_weight='transmission_weight', return_full_data=True)
    else:
        simResult = simAlg(graph, globalInfectionRate, recovery_rate, rho=0.0001,transmission_weight='transmission_weight', return_full_data=False)
    stop_b = time.time()

    record.print("simulation completed in {} seconds".format(stop_b - stop_a))
    record.print("total build and sim time: {}".format(stop_b-start))
    time_to_immunity = simResult[0][-1]
    final_uninfected = simResult[1][-1]
    final_recovered = simResult[3][-1]
    percent_uninfected = final_uninfected / (final_uninfected + final_recovered)
    record.last_runs_percent_uninfected = percent_uninfected
    record.print("The infection quit spreading after {} days, and {} of people were never infected".format(time_to_immunity,percent_uninfected))

    return simResult

def partitionOrdinals(groupsByCategory, partition_size, key):
    maximum = max(popsByCategory[key].keys())
    minimum = min(popsByCategory[key].keys())
    #partitioned_groups = partitionNames = (['{}:{}'.format(inf*partition_size, (inf+1)*partition_size) for inf in range(minimum//partition_size,maximum//partition_size)])
    #intNames = {inf :'{}:{}'.format(inf*partition_size, (inf+1)*partition_size) for inf in range(minimum//partition_size,maximum//partition_size)}
    partitioned_groups = [{key: '{}:{}'.format(i*partition_size, (i+1)*partition_size), 'list': []} for i in range(0, maximum//partition_size+1)]
    for i in groupsByCategory[key].keys():
        partitioned_groups[i//partition_size]['list'].extend(groupsByCategory[key][i])
    return partitioned_groups

def returnContactMatrix(graph, groups):
    print("WIP")
def partitionOrdinalsToDict(groupsByCategory, partition_size, key):
    maximum = max(popsByCategory[key].keys())
    minimum = min(popsByCategory[key].keys())
    partition_groups = []
    for i in groupsByCategory[key].keys():
        print("pause")

#def binarySearchGlobalInfectivity(R0, clusteringAlg, simAlg, transmissionWeighter)
record = Record()
record.print( "loading and sorting synthetic environment")
start = time.time()
#load people datasets
populace = loadPickles("people_list_serialized.pkl")
popsByCategory = sortPopulace(populace, ['sp_hh_id', 'work_id', 'school_id', 'race', 'age'])
age_grouped_pops = partitionOrdinals(popsByCategory, 5, 'age')
print("stop and CHECK YO ")
#load locations
loc_types = {"school": 0.3 , "workplace": 0.3, "household": 1}

#load location data from location files if necessary, otherwise load from csv
if False:
    locale = {}
    for type in loc_types:
        locale.update(loadPickles("{}s_list_serialized.pkl".format(type)))
        for location in locale:
            locale[location]["type"] = type# adds school, workplace, etc under 'type' key
            #locale[locale[location]['sp_id']] = locale[location] # makes the sp_id the key
            #locale.pop[location]
    df.columns = (df.loc['sp_id'][:])
    df.drop('sp_id')
    locale = df.to_dict()
    df.to_csv("./locale.csv")

#df = pd.read_csv("./locale.csv", index_col = 0)
locale = pickle.load(open("./locale.pkl", "rb"))
stop = time.time()
record.print("finished in {} seconds".format(stop - start))


#[t, S, I, R] = simulateGraph(clusterRandom2,workAvgDegree)
#plt.plot(t,I,label = 'random')
mask_scalar = 0.3
loc_weights = {"school_id": 0.1 , "work_id": 0.2, "sp_hh_id": 1}
weighter = TransmissionWeighter(loc_weights, mask_scalar)
weighter.record(record)

#----------------------------------------------------------------------
# SERIES OF RUNS with associated PLOTS
labels = []
sol = []
[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5],full_data = False)

sol.append([t,S,I,R])
labels.append('Uninfected count using Strogatz nets \nwith 50% random edges, control test')

[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.2])
sol.append([t,S,I,R])
labels.append('With 20% random Strogatz nets')

[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], exemption = 'schools')
sol.append([t,S,I,R])
labels.append('With primary schools closed')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces': 0.5, 'schools': 0.5})
sol.append([t,S,I,R])
labels.append('With 50% public masking')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces':0 , 'schools':1})
sol.append([t,S,I,R])
labels.append('With school masking')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces': 1, 'schools': 0})
sol.append([t,S,I,R])
labels.append('With workplace masking')

#----------------------------------------------
### PLOT RESULTS
nrows = 3
yl = ['S', 'I', 'R']

# Loop through the states S,I,R
for p in range(0,3):
    plt.subplot(nrows,1,p+1)
    # Loop through the simulations
    for i,label in enumerate(labels):
        plt.plot(sol[i][0], sol[i][p+1], label= label)
        plt.ylabel("# %s" % yl[p])
    if p == 1: plt.legend(loc = 1, prop={'size': 7}, framealpha=0.5)
    plt.xlabel("days")


#node_investigation = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.0001, transmission_weight ='transmission_weight',return_full_data = True)
#showGroupComparison(node_investigation, 'race', [1,2], popsByCategory)
#node_investigation.animate(popsByCategory['school_id'][450143554])

#if not nx.is_connected(graph):
#    print("warning: graph is not connected, there are {} components".format(nx.number_connected_components(graph.subgraph(popsByCategory['work_id'][505001334]))))
#node_investigation.animate()

### ADD COMMENTS

record.dump()
plt.savefig("./simResults/{}/plot".format(record.stamp))
plt.show()
#plt.plot(node_investigation.summary(popsByCategory['race'][3])[1]['I']/racePops[2],label = "infected students")
#plt.plot(node_investigation.summary(graph,label = "infected students")



