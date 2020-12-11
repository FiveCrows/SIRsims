import matplotlib.pyplot as plt
import pandas as pd
from  modelingToolkit import *
import seaborn as sns
import os 

#def getDegrees(model):
#    degrees = [len(graph[person]) for person in 

def plotContactMatrix(model, partitioner, env_indices, title = "untitled", ax = plt):
    '''
    This function plots the contact matrix for a structured environment
    :param p_env: must be a structured environment

    :param partitioner: Partitioner
        for specifying the partition of the contact matrix 
    
    :param

    '''
    
    contact_matrix = getContactMatrix(model,partitioner,env_indices)        
    ax.imshow(contact_matrix)
    
    plt.title("Contact Matrix for {}".format(title))
    labels = partitioner.labels
    axisticks= list(range(15))
    plt.xticks(axisticks, labels, rotation= 'vertical')
    plt.yticks(axisticks, labels)
    plt.xlabel('Age Group')
    plt.ylabel('Age Group')                

def getContactMatrix(model, partitioner, env_indices):
    n_sets = partitioner.num_sets
    cm = np.zeros([n_sets, n_sets])
    setSizes = np.zeros(n_sets)
    #add every
    for index in env_indices:
        env = model.environments[index]
        #assigns each person to a set
        placements, partition = partitioner.placeAndPartition(env.members, model.populace)
        setSizes += np.array([len(partition[index]) for index in partition])
        for edge in env.edges:
            cm[placements[edge[0]], placements[edge[1]]] += 1
    cm = np.nan_to_num([np.array(row)/setSizes for row in cm])
    return cm

def plotSIR(self, memberSelection = None):
    """
    For members of the entire graph, will generate three charts in one plot, representing the frequency of S,I, and R, for all nodes in each simulation
    """

    rowTitles = ['S','I','R']
    fig, ax = plt.subplots(3,1,sharex = True, sharey = True)
    simCount = len(self.sims)
    if simCount == []:
        print("no sims to show")
        return
    else:
        for sim in self.sims:
            title = sim[0]
            sim = sim[1]
            t = sim.t()
            ax[0].plot(t, sim.S())
            ax[0].set_title('S')

            ax[1].plot(t, sim.I(), label = title)
            ax[1].set_ylabel("people")
            ax[1].set_title('I')
            ax[2].plot(t, sim.R())
            ax[2].set_title('R')
            ax[2].set_xlabel("days")
    ax[1].legend()
    plt.show()

def getPeakPrevalences(self):
    return [max(sim[0].I()) for sim in self.sims]

#If a structuredEnvironment is specified, the partition of the environment is applied, otherwise, a partition must be passed
def plotBars(self, partitioner, env_indices, SIRstatus = 'R', normalized = False):
    """
    Will show a bar chart that details the final status of each partition set in the environment, at the end of the simulation
    :param environment: must be a structured environment
    :param SIRstatus: should be 'S', 'I', or 'R'; is the status bars will represent
    :param normalized: whether to plot each bar as a fraction or the number of people with the given status
    #TODO finish implementing None environment as entire graph
    """

    partition = partitioner
    for index in env_indices:
        if isinstance(environment, StructuredEnvironment):
            partitioned_people = environment.partition
            partition = environment.partitioner

        simCount = len(self.sims)
        partitionCount = partition.num_sets
        barGroupWidth = 0.8
        barWidth = barGroupWidth/simCount
        index = np.arange(partitionCount)

        offset = 0
        for sim in self.sims:
            title = sim[0]
            sim = sim[1]

            totals = []
            end_time = sim.t()[-1]
            for index in partitioned_people:
                set = partitioned_people[index]
                if len(set) == 0:
                    #no bar if no people
                    totals.append(0)
                    continue
                total = sum(status == SIRstatus for status in sim.get_statuses(set, end_time).values()) / len(set)
                if normalized == True:  total = total/len(set)
                totals.append[total]

            #totals = sorted(totals)
            xCoor = [offset + x for x in list(range(len(totals)))]
            plt.bar(xCoor,totals, barWidth, label = title)
            offset = offset+barWidth
    plt.legend()
    plt.ylabel("Fraction of people with status {}".format(SIRstatus))
    plt.xlabel("Age groups of 5 years")
    plt.show()
    plt.savefig(self.basedir+"/evasionChart.pdf")

def getR0(self):
    sim = self.sims[-1]
    herd_immunity = list.index(max(sim.I))
    return(self.population/sim.S([herd_immunity]))

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



