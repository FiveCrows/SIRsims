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
class TransmissionWeighter:
    def __init__(self, loc_scalars, mask_scalar):#, loc_masking):
        self.name = 'sole'
        self.global_weight = 1
        self.mask_scalar = mask_scalar
        self.loc_scalars = loc_scalars
        #self.loc_masking = loc_masking
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
        pass

    #WIP
    def record(self, record):
        record.print(str(self.__dict__))



class PopulaceGraph:
    def __init__(self, weighter, environment_degrees, environment_masking = None, graph = None, populace = None, pops_by_category = None, categories = ['sp_hh_id', 'work_id', 'school_id', 'race', 'age']):
        self.trans_weighter = weighter
        self.isBuilt = False
        self.record = Record()
        self.sims = []
        self.contactMatrix = None
        self.environment_degrees = environment_degrees
        self.environment_masking = environment_masking
        if graph == None:
            self.graph = nx.Graph()
        if populace == None:
        # for loading people objects from file
            with open("people_list_serialized.pkl", 'rb') as file:
                x = pickle.load(file)
            # return represented by dict of dicts
            self.populace = ({key: (vars(x[key])) for key in x})  # .transpose()
        else:
            self.populace = populace

        if pops_by_category == None:
        # for sorting people into categories
        # takes a dict of dicts to rep resent populace and returns a list of dicts of lists to represent groups of people with the same
        # attributes
            pops_by_category = {category: {} for category in categories}
            for person in self.populace:
                for category in categories:
                    try:
                        pops_by_category[category][self.populace[person][category]].append(person)
                    except:
                        pops_by_category[category][self.populace[person][category]] = [person]
            self.pops_by_category = pops_by_category
        else:
            self.pops_by_category = pops_by_category


    def clusterRandom(self, group, location, masking, params):
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
                    weight =  self.weighter.getWeight(edge[0],edge[1], location, masking)
                    self.graph.add_edge(edge[0],edge[1], transmission_weight = weight)

        else:
            for i in range(member_count-1):
                nodeA = group[i]
                for j in range(i+1,member_count):
                    if random.random()<edgeProb:
                        nodeB = group[j]
                        weight = self.weighter.getWeight(nodeA,nodeB,location)
                        self.graph.add_edge(nodeA,nodeB, transmission_weight = weight)

    #WIP
    def clusterPartitions(self, group, location, member_count, weighter, masking, params):
        partition_size = params[0]
        mixing_rate = params[1]
        if partition_size>member_count:
            self.clusterDense(group, member_count, masking, params)
            return
        #groups = nGroupAssign()


    def clusterDense(self, members, location, masking = None, params = None):
        member_count = len(members)
        #memberWeightScalar = np.sqrt(memberCount)
        for i in range(member_count):
            for j in range(i):
                weight = self.trans_weighter.getWeight(members[i], members[j], location, masking)
                self.graph.add_edge(members[i], members[j], transmission_weight = weight) #/ memberWeightScalar)


    def clusterStrogatz(self, members, location, masking, params):
        member_count = len(members)
        if member_count == 1:
            return

        local_k = params[0]
        rewire_p = params[1]
        if (local_k % 2 != 0):
            self.record.print("Error: local_k must be even")
            local_k = local_k+1
        if local_k >= member_count:
            self.clusterDense(members, location, masking)

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
                    weight = self.trans_weighter.getWeight(nodeA,nodeB, location, masking)
                self.graph.add_edge(nodeA, nodeB, transmission_weight=self.trans_weighter.getWeight(nodeA,nodeB, location, masking))


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


    def clusterGroups(self, environment, clusterAlg, masking, params=None):
        #self.record.print("clustering {} groups with the {} algorithm".format(classifier, clusterAlg.__name__))
        start = time.time()
        # # stats = {"classifier": }
        groups = self.pops_by_category[environment]
        group_count = len(groups)

        initial_weights = self.graph.size()
        for key in groups.keys():
            if key == None:
                continue
            group = groups[key]
            clusterAlg(group, environment, masking, params)

        weights_added = self.graph.size() - initial_weights
        stop = time.time()
        self.record.print("{} weights added for {} environments in {} seconds".format(weights_added, len(self.pops_by_category[environment].keys()), stop - start))


    def build(self, clusteringAlg, params=None, exemption=None, masking = {'schools': None, 'workplaces': None}):
        self.record.print('\n')
        self.record.print("building populace into graphs with the {} clustering algorithm".format(clusteringAlg.__name__))
        start = time.time()
        self.graph = nx.Graph()

        #dense cluster for each household
        self.clusterGroups('sp_hh_id', self.clusterDense, None)
        #cluster schools and workplaces with specified clustering alg
        if exemption != 'workplaces':
            self.clusterGroups('work_id', self.clusterStrogatz, self.environment_masking['work'], [self.environment_degrees['work'], 0.5])
        if exemption != 'schools':
            self.clusterGroups('school_id', self.clusterStrogatz,self.environment_masking['school'], [self.environment_degrees['school'], 0.5])
        stop_a = time.time()
        self.record.print("Graph completed in {} seconds.".format((stop_a - start)))


    def partitionOrdinals(self, key, partition_size):
        maximum = max(self.pops_by_category[key].keys())
        minimum = min(self.pops_by_category[key].keys())
        # partitioned_groups = partitionNames = (['{}:{}'.format(inf*partition_size, (inf+1)*partition_size) for inf in range(minimum//partition_size,maximum//partition_size)])
        # intNames = {inf :'{}:{}'.format(inf*partition_size, (inf+1)*partition_size) for inf in range(minimum//partition_size,maximum//partition_size)}
        partitioned_groups = [{key: '{}:{}'.format(i * partition_size, (i + 1) * partition_size), 'list': []} for i in
                              range(0, maximum // partition_size + 1)]

        for i in self.pops_by_category[key].keys():
            partitioned_groups[i // partition_size]['list'].extend(self.pops_by_category[key][i])

        partition_count = partitioned_groups.__len__()
        self.id_to_partition = {}
        for partition in range(partition_count):
            list = partitioned_groups[partition]['list']
            for id in list:
               self.id_to_partition[id] = partition
        return partitioned_groups


    def constructContactMatrix(self, key, partition_size, reshape = None):
        partitioned_groups = self.partitionOrdinals(key, partition_size)

        #this is here in case the loaded matrix has unusual shape, like combining all ages  75+ into one group
        #loops back from the partitions at the end
        if reshape != None:
            for i in range(len(partitioned_groups)-1, reshape-1, -1):
                partitioned_groups[reshape - 1]['list'] = partitioned_groups[reshape - 1]['list'] + partitioned_groups[i]['list']
                del partitioned_groups[i]

        partition_count = partitioned_groups.__len__()
        partition_sizes = np.empty(partition_count)
        contact_matrix = np.empty([partition_count, partition_count])

        #create dict to associate each id to partition
        id_to_partition = {}
        for partition in range(partition_count):
            list = partitioned_groups[partition]['list']
            partition_sizes[partition] = list.__len__()
            for id in list:
               id_to_partition[id] = partition

        for i in self.graph:
            iPartition = id_to_partition[i]
            for j in self.graph[i]:
                jPartition = id_to_partition[j]
                contact_matrix[iPartition, jPartition] += self.graph[i][j]['transmission_weight']/partition_sizes[iPartition]
        plt.imshow(np.array([ row/np.linalg.norm(row) for row in contact_matrix]))
        self.contact_matrix = contact_matrix


    def fitGraphToContactMatrix(self, contact_matrix, key, partition_size):
        assert contact_matrix.shape[0] == contact_matrix.shape[1], "contact matrices must be symmetric"
        self.constructContactMatrix(key, partition_size, reshape = contact_matrix.shape[0])
        assert self.contact_matrix.shape == contact_matrix.shape, "mismatch contact matrix shapes"

        scaleMatrix = (self.contact_matrix + self.contact_matrix.transpose()) / (contact_matrix + contact_matrix.transpose())
        for i in self.graph:
            for j in self.graph[i]:
                scalar = scaleMatrix[self.id_to_partition[i], self.id_to_partition[j]]
                self.graph[i][j]['transmission_weight'] = self.graph[i][j]['transmission_weight'] * scalar


    #given a  list of lists to partition N_i, the nodes in a graph, this function produces a 2d array,
    #contact_matrix, where contact_matrix[i,j] is the sum total weight of edges between nodes in N_i and N_j, divided by number of nodes in N_i


    def simulate(self, gamma, tau, simAlg = EoN.fast_SIR,  full_data = False):
        start = time.time()
        simResult = simAlg(self.graph, gamma, tau, rho=0.0001, transmission_weight='transmission_weight', return_full_data=full_data)
        stop = time.time()
        self.record.print("simulation completed in {} seconds".format(stop - start))
        time_to_immunity = simResult[0][-1]
        final_uninfected = simResult[1][-1]
        final_recovered = simResult[3][-1]
        percent_uninfected = final_uninfected / (final_uninfected + final_recovered)
        #self.record.last_runs_percent_uninfected = percent_uninfected
        self.record.print("The infection quit spreading after {} days, and {} of people were never infected".format(time_to_immunity,percent_uninfected))
        self.sims.append(simResult)

    def plotSIR(self):
        rowTitles = ['S','I','R']
        fig, ax = plt.subplots(3,1)
        simCount = len(self.sims)
        if simCount == []:
            print("no sims to show")
            return
        else:
            for i in range(simCount):
                print("just test here for now")

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

