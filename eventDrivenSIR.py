import numpy as np
import scipy as sci
import networkx as nx
import pandas as pd
import random
import itertools
epidemicSims = 10
houseHolds = 600
houseHoldSize = 4
communitySize = houseHoldSize * houseHolds
schoolGroupSize = 20
workGroupSize = 10
employmentRate = 0.9
recoveryRate = 1
globalInfectionRate = 1
houseInfectivity = .1
workClasses = ['default', 'unemployed', 'school']
workInfectivity = .05

ageGroups = [[0,5], [5,8], [18,65], [65,90]]
ageGroupWeights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
everyone = pd.DataFrame(data = [{'index' : i} for i in range(communitySize)])
#incomeGroups =
#incomeGroupWeights =

tau = 1 #transmission factor
gamma = 1 #recovery rate
initial_infected = 1


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

#a function which returns a list of tuples randomly assigning nodes to groups of size probability n
def p_nGroupAssign(members, p_n):
    length = len(members)
    random.shuffle(members)
    pos = 0
    groupNumber = 0
    dict = {}
    while True:
        groupSize = random.choices(range(len(p_n)), weights=p_n)[0]+1
        if(pos+groupSize>length):
            dict[groupNumber] = (itertools.islice(members, pos, pos + groupSize))
            break
        dict[groupNumber] = list(itertools.islice(members, pos, pos + groupSize))
        groupNumber = groupNumber + 1
        pos = pos+groupSize
    return dict

def p_attributeAssign(members, attributes, probabilites):
    random.shuffle(members)
    for member in members:
        node["ageGroup"] = random.choices(ageGroups, weights = ageGroupWeights)[0]


#TODO connect list of groups with weight
def updateGraph(graph, groups, groupWeight):
    for key in groups.keys():
        groupSize = len(groups[key])
        for i in range(groupSize):
            for j in range(i):
                graph.add_edge(groups[key][i],groups[key][j] ,transmission_weight = groupWeight)


# assign people to households
houseMap = p_nGroupAssign(list(range(communitySize)), [0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

#idea: weight age groups to represent common household distribution, such as parents in same age group, + children
#def assignAges(everyone, houseGroups):
for node in everyone():
    node["ageGroup"] = random.choices(ageGroups, weights = ageGroupWeights)[0]


#TODO assign households and nodes in households to neighborhoods:
#Idea: track neighborhoods and whole city as 'global groups', groups which occur global infections to eachother, but aren't necessarily bigger risks because the group is bigger
#these are contacts that occur between strangers who possibly share the same transit, gym, etc.

#TODO a function that returns the infection rate of neighborhoods and or city

#TODO assign children to schoolGroups by neighborhood

#TODO write function which takes a weight matrix, a small probability of initial infected, and performs the gillespie algorithm including risk from global contact,
#and returns  t as a 1-d array, and S, I, R the states of each individual at each time


#TODO a function that will filter S,I,R to only subsets of nodes, picked by conditions


#TODO write a function to condense S,I,R into 1-d arrays describing the portion at given times


#TODO write a function to spline 1d t,S,I,R arrays into even time intervals so that multiple runs can be averaged


#TODO a function to draw a sparse weights matrix in a graph

