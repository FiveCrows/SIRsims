import matplotlib.pyplot as plt
import pandas as pd
from  modelingToolkit import *
import seaborn as sns
import os 

#def getDegrees(model):
#    degrees = [len(graph[person]) for person in 
    
def getDegreeHistogram(model, env_indexes, normalized = True): 
    """
    :param model: PopulaceGraph
    Produce a histogram of the populace 
    :param normalized: normalize the histogram if true. 
    """
    degreeCounts = [0] * 100
    for index in env_indexes:
        env = model.environments[index]
        people = env.members
        graph = model.graph.subgraph(people)
        
        for person in people:
            try:
                degree = len(graph[person])
            except:
                degree = 0
            degreeCounts[degree] += 1
    while degreeCounts[-1] == 0:
        degreeCounts.pop()
    return degreeHistogram

def aFormatGraph(model, folder):
    ageGroups = [[0,5], [5,18], [18,50], [50,65], [65,100]]
    enumerator = {}
    try:
        os.mkdir(folder)
    except:
        pass
    for i, group in enumerate(ageGroups):
        enumerator.update({j:i for j in range(group[0], group[1])})    
    open(folder +"/nodes.txt","a").writelines(["{} {}\n".format(item,enumerator[model.populace[item]['age']])  for item in model.graph.nodes()])
    with  open(folder +"/edges.txt","a") as file:
        adj = model.graph.adj
        for edgeA in adj:            
            for edgeB in adj[edgeA]:
                file.writelines("{} {} {}/n".format(edgeA,edgeB, adj[edgeA][edgeB]['transmission_weight']))
                file.writelines("{} {} {}/n".format(edgeB,edgeA, adj[edgeA][edgeB]['transmission_weight']))
            
