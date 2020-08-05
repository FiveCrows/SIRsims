import random
from os import mkdir
import EoN
import networkx as nx
import itertools
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import json

class TransmissionWeighter:
    def __init__(self, env_scalars, mask_scalar, env_masking, name ='default'):#, loc_masking):
        self.name = name
        self.global_weight = 1
        self.mask_scalar = mask_scalar
        self.env_masking = env_masking
        self.env_scalars = env_scalars

        #self.loc_masking = loc_masking
        #self.age_scalars = age_scalars

    def getWeight(self, personA, personB, env, masking):
        masking = self.env_masking[env]
        weight = self.global_weight
        try:
            weight = weight*self.env_scalars[env]
        except:
            print("locale type not identified")

        if (masking != 0):
            if random.random()<masking:
                weight = weight*self.mask_scalar
        if masking != None:
            if random.random()<masking:
                weight = weight*self.mask_scalar
        return weight

    #WIP
    def reweight(graph, groups):
        pass

    #WIP
    def record(self, record):
        record.print(str(self.__dict__))

    def __str__(self):
        string = "\n Weighter name: {} \n".format(self.name)
        string += "Global default weight: {} \n".format()
        string +="Mask risk reduction scalar: {} \n".format(self.mask_scalar)
        string +="Environment weight scalars: \n"
        string += json.dumps(self.env_scalars)
        return string


class PopulaceGraph:
    def __init__(self, weighter, environment_degrees, environment_masking =  {'work': 0, 'school':0}, graph = None, populace = None, pops_by_category = None, categories = ['sp_hh_id', 'work_id', 'school_id', 'race', 'age'], slim = False):
        self.trans_weighter = weighter
        self.isBuilt = False
        self.record = Record()
        self.sims = []
        self.contactMatrix = None
        self.environment_degrees = environment_degrees
        self.environment_masking = environment_masking
        self.total_weight = 0
        if graph == None:
            self.graph = nx.Graph()

        #load populace from file if necessary
        if populace == None:
        # for loading people objects from file
            with open("people_list_serialized.pkl", 'rb') as file:
                x = pickle.load(file)

            # return represented by dict of dicts
        if slim == False:
            self.populace = ({key: (vars(x[key])) for key in x})  # .transpose()
        else:
            self.populace = {}
            for key in x:
                if random.random()>0.9:
                    self.populace[key] = (vars(x[key]))


        if pops_by_category == None:
        # for sorting people into categories
        # takes a dict of dicts to rep resent populace and returns a list of dicts of lists to represent groups of people with the same
        # attributes
            env_rename = {'sp_hh_id': 'household'}
            pops_by_category = {category: {} for category in categories}
            #pops_by_category{'populace'} = []
            for person in self.populace:
                for category in categories:
                    try:
                        pops_by_category[category][self.populace[person][category]].append(person)
                    except:
                        pops_by_category[category][self.populace[person][category]] = [person]
            self.pops_by_category = pops_by_category
        else:
            self.pops_by_category = pops_by_category

    def clusterRandom(self, group, env, masking, params):
        member_count = len(group)
        avg_degree = params
        if avg_degree >= member_count:
            self.clusterDense(self.graph, group, member_count, self.weighter, params)
            return
        edgeProb = 2 * avg_degree / (member_count - 1)

        if member_count < 100:  # otherwise this alg is too slow
            total_edges = avg_degree * member_count
            pos_edges = itertools.combinations(group,2)
            for edge in pos_edges:
                if random.random()<edgeProb:
                    weight =  self.weighter.getWeight(edge[0], edge[1], env, masking)
                    self.graph.add_edge(edge[0], edge[1], transmission_weight = weight, environment = env)
                    self.total_weight += weight
        else:
            for i in range(member_count-1):
                nodeA = group[i]
                for j in range(i+1,member_count):
                    if random.random()<edgeProb:
                        nodeB = group[j]
                        weight = self.weighter.getWeight(nodeA, nodeB, env)
                        self.graph.add_edge(nodeA, nodeB, transmission_weight = weight, environment = env)

    #WIP
    def clusterPartitions(self, group, location, member_count, weighter, masking, params):
        partition_size = params[0]
        mixing_rate = params[1]
        if partition_size>member_count:
            self.clusterDense(group, member_count, masking, params)

            return
        #groups = nGroupAssign()


    def clusterDense(self, members, env, masking = None, params = None):
        member_count = len(members)
        #memberWeightScalar = np.sqrt(memberCount)
        for i in range(member_count):
            for j in range(i):
                weight = self.trans_weighter.getWeight(members[i], members[j], env, masking)
                self.graph.add_edge(members[i], members[j], transmission_weight = weight, environment = env) #/ memberWeightScalar)
                self.total_weight +=weight

    def clusterStrogatz(self, members, env, masking, params):
        member_count = len(members)
        if member_count == 1:
            return

        local_k = params[0]
        rewire_p = params[1]
        if (local_k % 2 != 0):
            self.record.print("Error: local_k must be even")
            local_k = local_k+1
        if local_k >= member_count:
            self.clusterDense(members, env, masking)

        for i in range(member_count):
            nodeA = members[i]
            for j in range(1, local_k // 2+1):
                if j == 0:
                    continue
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = members[(i + random.choice(range(member_count - 1))) % member_count]

                else:
                    nodeB = members[(i + j) % member_count]
                weight = self.trans_weighter.getWeight(nodeA, nodeB, env, masking)
                self.total_weight+=weight
                self.graph.add_edge(nodeA, nodeB, transmission_weight=weight, environment = env)

    #needs to be updated
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

    #WIP
    def clusterGroupsByPA(graph, groups):
        for key in groups.keys():
            memberCount = len(groups[key])


    def clusterGroups(self, env, clusterAlg, masking, params=None):
        #self.record.print("clustering {} groups with the {} algorithm".format(classifier, clusterAlg.__name__))
        start = time.time()
        # # stats = {"classifier": }
        env_to_category = {"household": "sp_hh_id","work":"work_id","school": "school_id"}
        groups = self.pops_by_category[env_to_category[env]]
        group_count = len(groups)

        initial_weights = self.graph.size()
        for key in groups.keys():
            if key == None:
                continue
            group = groups[key]
            clusterAlg(group, env, masking, params)

        weights_added = self.graph.size() - initial_weights
        stop = time.time()
        self.record.print("{} weights added for {} environments in {} seconds".format(weights_added, len(self.pops_by_category[env_to_category[env]].keys()), stop - start))


    def build(self, clusteringAlg, params=None, exemption=None, masking = {'schools': None, 'workplaces': None}):
        self.record.print('\n')
        self.record.print("building populace into graphs with the {} clustering algorithm".format(clusteringAlg.__name__))
        start = time.time()
        self.graph = nx.Graph()

        #dense cluster for each household
        self.clusterGroups('household', self.clusterDense, None)
        #cluster schools and workplaces with specified clustering alg
        if exemption != 'workplaces':
            self.clusterGroups('work', self.clusterStrogatz, self.environment_masking['work'], [self.environment_degrees['work'], 0.5])
        if exemption != 'schools':
            self.clusterGroups('school', self.clusterStrogatz,self.environment_masking['school'], [self.environment_degrees['school'], 0.5])
        stop_a = time.time()
        #self.record.print("Graph completed in {} seconds.".format((stop_a -


    def partition(self, people, attribute, enumerator, partition_limit = 16):
        partition =  [[] for i in range(len(np.unique(list(enumerator.values()))))] #yup... this creates an empty list for every partition element in a list,
        #A better way? IDK ...

        id_to_element = {}
        for person in people:
            element = enumerator[self.populace[person][attribute]]
            partition[element].append(person)
            id_to_element[person] = element

        return partition, id_to_element

    def sumAllWeights(self):
        sum = 0
        for i in self.graph:
            for j in self.graph[i]:
                sum = sum+self.graph[i][j]['transmission_weight']
        self.entire_weight_sum = sum
        return sum

    #def construct_weight_matrix(partition, id_to_partition):

    def constructWeightMatrix(self, partition, id_to_partition):
        weights = np.zeros([len(partition), len(partition)])
        for id in id_to_partition:
            iPartition = id_to_partition[id]
            for j in self.graph[id]:
                jPartition = id_to_partition[j]
                weights[iPartition, jPartition] += self.graph[id][j]['transmission_weight']
        #plt.imshow(np.array([row / np.linalg.norm(row) for row in contact_matrix]))
        return weights

    def constructEdgeMatrix(self, partition, id_to_partition):
        weights = np.zeros([len(partition), len(partition)])
        for id in id_to_partition:
            iPartition = id_to_partition[id]
            for j in self.graph[id]:
                jPartition = id_to_partition[j]
                edges[iPartition, jPartition] += 1
        #plt.imshow(np.array([row / np.linalg.norm(row) for row in contact_matrix]))
        return edges

    def partitionToContactMatrix(self, partition, id_to_partition):
        element_sizes = [len(element) for element in partition]
        partition_elements = len(partition)
        weight_matrix = self.constructWeightMatrix(partition, id_to_partition)
        contact_matrix = np.zeros([partition_elements, partition_elements])
        for i in range(partition_elements):
            for j in range(partition_elements):
                contact_matrix[i,j] = weight_matrix[i,j]/element_sizes[i]
        return contact_matrix

    def partitionToPreferenceMatrix(self, partition, id_to_partition):
        element_sizes = [len(element) for element in partition]
        partition_elements = len(partition)
        number_people = len(id_to_partition.keys())
        weight_matrix = self.constructWeightMatrix(partition, id_to_partition)
        cumulative_weight = sum(sum(weight_matrix))
        preference_matrix = np.zeros([partition_elements, partition_elements])
        cumulative_pos_edges = number_people*(number_people-1)/2
        for i in range(partition_elements):
            for j in range(i, partition_elements):
                if i == j:
                    pos_edges = element_sizes[i]*(element_sizes[i] -1)/2
                else:
                    pos_edges = element_sizes[i]*element_sizes[j]
                preference_matrix[i,j] = (weight_matrix[i,j]/cumulative_weight)*(cumulative_pos_edges/pos_edges)
                preference_matrix[j,i] = preference_matrix[i,j]
        plt.imshow(preference_matrix)
        plt.show()
        return preference_matrix



    def fitWithContactMatrix(self, people, contact_matrix, key, partition_size, show_scale = False):
        assert contact_matrix.shape[0] == contact_matrix.shape[1], "contact matrix must be square"
        assert contact_matrix.shape == contact_matrix.shape, "mismatch contact matrix shapes"

        entireGraphWeight = self.sumAllWeights()

        scaleMatrix = (contact_matrix + contact_matrix.transpose()) / (contact_matrix + contact_matrix.transpose())
        for i in self.graph:
            for j in self.graph[i]:
                scalar = scaleMatrix[self.id_to_partition[i], self.id_to_partition[j]]
                self.graph[i][j]['transmission_weight'] = self.graph[i][j]['transmission_weight'] * scalar

        renormFactor = entireGraphWeight/self.sumAllWeights()
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j]['transmission_weight'] = self.graph[i][j]['transmission_weight'] * renormFactor

        if show_scale:
            plt.imshow(scaleMatrix*renormFactor)
            plt.title("scale matrix")
            plt.show()
            plt.savefig("./simResults/{}/NodeDegreePlot".format(self.record.stamp))
    #given a  list of lists to partition N_i, the nodes in a graph, this function produces a 2d array,
    #contact_matrix, where contact_matrix[i,j] is the sum total weight of edges between nodes in N_i and N_j, divided by number of nodes in N_i


    def simulate(self, gamma, tau, simAlg = EoN.fast_SIR, title = None, full_data = True):
        start = time.time()
        simResult = simAlg(self.graph, gamma, tau, rho=0.0001, transmission_weight='transmission_weight', return_full_data=full_data)
        stop = time.time()
        self.record.print("simulation completed in {} seconds".format(stop - start))

        #doesn't work returning full results
        #time_to_immunity = simResult[0][-1]
        #final_uninfected = simResult[1][-1]
        #final_recovered = simResult[3][-1]
        #percent_uninfected = final_uninfected / (final_uninfected + final_recovered)
        #self.record.last_runs_percent_uninfected = percent_uninfected
        #self.record.print("The infection quit spreading after {} days, and {} of people were never infected".format(time_to_immunity,percent_uninfected))
        self.sims.append([simResult, title])

    #doesn't work, outdated
    def plotContactMatrix(self, key, partition_size):
        self.constructContactMatrix(key, partition_size)
        plt.imshow(self.contact_matrix)
        plt.show()
        plt.savefig("./simResults/{}/contactMatrix".format(self.record.stamp))

    def plotNodeDegreeHistogram(self):
        plt.hist([degree[1] for degree in nx.degree(self.graph)], 'auto')
        plt.ylabel("total people")
        plt.xlabel("degree")
        plt.show()
        plt.savefig("./simResults/{}/".format(self.record.stamp))

    def plotSIR(self):
        rowTitles = ['S','I','R']
        fig, ax = plt.subplots(3,1,sharex = True, sharey = True)
        simCount = len(self.sims)
        if simCount == []:
            print("no sims to show")
            return
        else:
            for sim in self.sims:
                title = sim[1]
                sim = sim[0]
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

    def plotEvasionChart(self, partition = None):
        for sim in self.sims:
            title = sim[1]
            sim = sim[0]

            totals = []
            end_time = sim.t()[-1]
            for element in partition:
                totals.append(sum(status == 'S' for status in sim.get_statuses(element, end_time).values())/len(element))
            #totals = sorted(totals)
            plt.bar(list(range(len(totals))),totals,label = title)
        plt.legend()
        plt.show()
        plt.ylabel("Fraction Uninfected")
        plt.xlabel("partition number ")
        plt.savefig("./simResults/{}/evasionChart".format(self.record.stamp))

    def __str__(self):
        string = "Model: "
        string += "\n Weighter Name: {}".format(self.trans_weighter.name)
        string += "\n {Node count: {}, Edge count: {}}".format(self.graph.number_of_nodes, self.graph.number_of_edges())


class Record:
    def __init__(self):
        self.log = ""
        self.comments = ""
        self.stamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.graph_stats = {}
        self.last_runs_percent_uninfected = 1
        mkdir("./simResults/{}".format(self.stamp))
    def print(self, string):
        print(string)
        self.log+=('\n')
        self.log+=(string)

    def addComment(self):
        comment = input("Enter comment")
        self.comments += comment
        self.log +=comment

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
        log_txt = open("./simResults/{}/log.txt".format(self.stamp), "w+")
        log_txt.write(self.log)
        if self.comments != "":
            comment_txt = open("./simResults/{}/comments.txt".format(self.stamp),"w+")
            comment_txt.write(self.comments)

