import matplotlib.pyplot as plt
import pandas as pd
from  modelingToolkit import *
import seaborn as sns


def getDegrees(model):
    degrees = [len(graph[person]) for person in 
    
def getDegreeHistogram(model, env_indexes, normalized = True): 
    """
    :param model: PopulaceGraph
    the PopulaceGraph model to produce histogram for
    :param normalized, when true the histogram will normalized
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

