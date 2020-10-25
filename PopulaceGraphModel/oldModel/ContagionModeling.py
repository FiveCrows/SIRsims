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
import math


class TransmissionWeighter:
    def genMaskScalar(self, mask_p):
        if random.random() < mask_p:
            mask_scalar = self.trans_weighter.mask_scalar
            if random.random() < (mask_p * mask_p):
                mask_scalar = self.trans_weighter.mask_scalar
            else:
                mask_scalar = 1
        return 1

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

class Partition:
    def __init__(self, members, populace, attribute, enumerator, contact_matrix, group_names = None):
        self.members = members
        self.attribute = attribute
        self.enumerator = enumerator
        self.num_groups = len(enumerator)
        self.group_names = group_names
        self.contact_matrix = contact_matrix
        self.partition = dict.fromkeys(set(enumerator.values()))
        self.id_to_partition = dict.fromkeys(members)

        for person in members:
            #determine the number for which group the person belongs in, depending on their attribute
            group = enumerator[populace[person][attribute]]
            #add person to  to dict in group
            self.partition[group].append(person)

        for group in self.partition:
            for person in group:
                self.id_to_partition[person]= group

    pass

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
            with open("../people_list_serialized.pkl", 'rb') as file:
                x = pickle.load(file)

            # return represented by dict of dicts
        if slim == False:
            self.populace = ({key: (vars(x[key])) for key in x})  # .transpose()
        else:
            self.populace = {}
            for key in x:
                if random.random()>0.9:
                    self.populace[key] = (vars(x[key]))
        self.population = len(self.populace)

        if pops_by_category == None:
        # for sorting people into categories
        # takes a dict of dicts to rep resent populace and returns a list of dicts of lists to represent groups of people with the same
        # attributes

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

    #wip, secondary level gettin a lil too wild, I'ma try a different way, one that won't be so slow
    # for implementing MGPA,
    def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, id_to_partition, mask_p, weight):
        w = self.trans_weighter.genMaskScalar(mask_p) * weight
        self.graph.add_edge(nodeA, nodeB, transmission_weight  = w)
        groupA = id_to_partition[nodeA]
        groupB = id_to_partition[nodeB]


        #grow secondary list
        #Adding B's friends to A's secondary
        for key in attachments[nodeA]["secondary"][nodeB]:
            attachments[nodeA]["secondary"][key].extend(attachments[nodeB]["secondary"][key])
        #Adding A's friends to B's secondary
        for key in attachments[nodeB]["secondary"][nodeA]:
            attachments[nodeB]["secondary"][key].extend(attachments[nodeA]["secondary"][key])

        #Adding B as secondary to A's friends
        for key in attachments[nodeA]:
            pass
        #Adding A as secondary to B's friends

            # grow primary list,
            # adding B to A, A to B
        attachments[nodeA]["primary"][groupB].append(nodeB)
        attachments[nodeB]["primary"][groupA].append(nodeA)

        #Adding A's friends to B

        #Adding B to A's friends
        #Adding A to B's friends
        #try:
            #attachments[""]
    #def clusterFromMatrix(self,group, env, masking, params):
    def clusterMatrixGuidedPreferentialAttachment(self, group, M, partition, id_to_partition, avg_contacts, r,mask_p):

        cumulative_weight = sum(M)
        num_people = len(group)
        weight = cumulative_weight / num_people
        total_pos_edges = num_people * (num_people - 1) / 2
        total_edges = num_people * avg_contacts
        random_edges = r * total_edges
        remaining_edges = total_edges - random_edges

        vecM = np.matrix.flatten(M)
        num_partitions = len(vecM)
        partitionAttachments = {}

        #don't forget to fix
        #speeds up in case there aren't many duplicates likely anyways
        random_duplicate_rate = (random_edges-1)/total_pos_edges
        # above, so much cancelled... hows that for some prob?
        if random_duplicate_rate > 0.01:
            rand_edges = random.choices(list(itertools.combinations(group,2)), k = random_edges)
            for edge in rand_edges:
                self.graph.add_edge(edge[0], edge[1],
                                    transmission_weight=self.trans_weighter.genMaskScalar(mask_p) * weight)
                try:
                    #partitionAttachments[]
                    pass
                except:
                    pass
        else:
            for i in range(random_edges):
                sel_A = random.choice(num_people)
                sel_B = (sel_A + random.choice(num_people-1))%num_people
                self.graph.add_edge(group[sel_A], group[sel_B],transmission_weight=self.trans_weighter.genMaskScalar(mask_p) * weight)

        #now adding preferential attachment edges
        partition_dist = [sum(vecM[:i] for i in range(num_partitions))]/sum(vecM)
        #partition_dist projects the edge_partition to  [0,1], such that the space between elements is in proportion to
        #the elements contact
        for i in range(remaining_edges):
            #this selects a partition element using partition_dist
            #then, from vec back to row/col
            selector = random.random()
            raw_partition = list(filter(range(num_partitions), lambda i: partition_dist[i]<(selector) & partition_dist[i+1]>(selector)))
            partition_A = raw_partition % M.shape[0]
            partition_B = raw_partition // M.shape[0]

    #WIP
    def clusterMatrixGuidedPreferentialAttachment2(self, group, M, partition, id_to_partition, avg_contacts, r, mask_p):
        cumulative_weight = sum(M)
        num_people = len(group)
        weight = cumulative_weight / num_people
        total_pos_edges = num_people * (num_people - 1) / 2
        total_edges = num_people * avg_contacts
        random_edges = r * total_edges
        remaining_edges = total_edges - random_edges

        for member in group:
            partition = id_to_partition[member]


    def clusterRandom(self, group, masking, avg_degree = None, num_edges = None, env = None, weight = None):
        member_count = len(group)
        #so the user can input avg_degree OR num_edges
        if avg_degree == None:
            assert(num_edges != None)
            avg_degree = num_edges/member_count

        if avg_degree > member_count-1:
            self.clusterDense(self.graph, member_count, self.weighter, group)
            return
        edgeProb = avg_degree / (member_count - 1)

        if member_count < 100:  # otherwise this alg is too slow
            total_edges = avg_degree * member_count
            pos_edges = itertools.combinations(group,2)
            for edge in pos_edges:
                if random.random()<edgeProb:
                    weight =  self.weighter.getWeight(edge[0], edge[1], env, masking)
                    if env == None:
                        assert(weight != None)
                    else:
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


    def clusterDense(self, members, masking=None, env=None, w = None):
        member_count = len(members)
        #memberWeightScalar = np.sqrt(memberCount)
        for i in range(member_count):
            for j in range(i):
                #originally weight was picked for the environment, but in order to
                # implement clustering by matrix I've updated to pass weight explicitly
                if env == None:
                    weight = w
                else:
                    weight = self.trans_weighter.getWeight(members[i], members[j], env, masking)

                self.graph.add_edge(members[i], members[j], transmission_weight = weight, environment = env) #/ memberWeightScalar)
                self.total_weight += weight


    def clusterStrogatz(self, members, masking, params, env = None, weight = None, num_edges = None):

        #unpack params
        local_k = params[0]
        rewire_p = params[1]
        # if only one person, don't bother
        member_count = len(members)
        if member_count == 1:
            return

        remainder = 0
        #if user is specifying a number of edges
        if num_edges != None:
            local_k = math.floor(num_edges*2/member_count)
            remainder = num_edges - local_k*member_count/2
            random.sample(members,remainder)

        if (local_k % 2 != 0):
            self.record.print("Error: local_k must be even")
            local_k = local_k+1
        if local_k >= member_count:
            self.clusterDense(members, masking, env=env, weight = weight)
            return

        for i in range(member_count):
            nodeA = members[i]

            for j in range(1, local_k // 2+1):
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = members[(i + random.choice(range(member_count - 1))) % member_count]
                else:
                    nodeB = members[(i + j) % member_count]

                weight = self.trans_weighter.getWeight(nodeA, nodeB, env, masking)
                self.total_weight+=weight
                self.graph.add_edge(nodeA, nodeB, transmission_weight=weight, environment=env)

            def clusterBipartite(self, members_A, members_B, edge_count, weight, p_random=0.1):
                # reorder groups by size
                A = min(members_A, members_B, key=len)
                if A == members_A:
                    B = members_B
                else:
                    B = members_A
                size_A = len(A)
                size_B = len(B)

                if members_A * members_B > edge_count:
                    print("warning, not enough possible edges for cluterBipartite")

                # distance between edge groups
                separation = int(math.ceil(size_B / size_A))

                # size of edge groups and remaining edges
                k = edge_count // size_A
                remainder = edge_count % size_A
                p_random = max(0, p_random - remainder / edge_count)

                for i in range(size_A):
                    begin_B_edges = (i * separation - k // 2) % size_B

                    for j in range(k):
                        if random.random() > p_random:
                            B_side = (begin_B_edges + j) % size_B
                            self.graph.addEdge(A[i], B[B_side], transmission_weight=weight)
                        else:
                            self.graph.addEdge(random.choice(A), random.choice(B))

                for i in range(remainder):
                    self.graph.addEdge(random.choice(A), random.choice(B))

            #don't for get to add remainder edges too

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

    #wip
    def clusterByContact(self, partition, matrix, masking, avg_contacts):
        totalContact = matrix.sum()
        totalPeople = sum([len(partition[group]) for group in partition])
        totalEdges = avg_contacts*totalPeople
        default_weight = totalContact/totalEdges

        for groupA in partition:
            for groupB in partition:
                weightSum = matrix[groupA,groupB]
                number_edges = int(weightSum/default_weight)
                w = weightSum/number_edges # a slight rescale to compensate rounding
                if groupA == groupB:
                    num_people = len(groupA)
                    k = number_edges//(2*num_people)
                    self.clusterStrogatz(groupA, masking, [0.1,None], weight = w, num_edges = number_edges)


                else:
                    self.clusterBipartite(groupA,groupB, number_edges, w)
        #for partition in
        #single list of group is to create a strogatz ring

    def getInfectionProb(self,w,beta,gamma):
        return w*beta/(gamma+w*beta)

    def returnNextGenMatrix(self, partition, id_to_partition, beta, gamma):
        #initiialize next gen matrix
        N = np.zeros([len(partition), len(partition)])

        #for each ij, add the probability of infection for each weight divided by members in i
        for id in id_to_partition:
            iPartition = id_to_partition[id]
            iMemberCount = len(partition[iPartition])
            for j in self.graph[id]:
                jPartition = id_to_partition[j]
                weight = self.graph[id][j]['transmission_weight']
                N[iPartition, jPartition] += self.getInfectionProb(weight,beta,gamma)/iMemberCount
        return N

    def clusterGroups(self, environment, clusterAlg, masking, params=None, weight = None):
        #self.record.print("clustering {} groups with the {} algorithm".format(classifier, clusterAlg.__name__))
        start = time.time()
        # # stats = {"classifier": }
        env_to_category = {"household": "sp_hh_id", "work": "work_id", "school": "school_id"}
        groups = self.pops_by_category[env_to_category[environment]]
        group_count = len(groups)

        initial_weights = self.graph.size()
        for key in groups.keys():
            if key == None:
                continue
            group = groups[key]
        clusterAlg(group, environment,  masking, params, w = weight)

        weights_added = self.graph.size() - initial_weights
        stop = time.time()
        self.record.print("{} weights added for {} environments in {} seconds".format(weights_added, len(self.pops_by_category[env_to_category[environment]].keys()), stop - start))


    def build(self, clusteringAlg, params=None, exemption=None, masking = {'schools': None, 'workplaces': None}):
        self.record.print('\n')
        self.record.print("building populace into graphs with the {} clustering algorithm".format(clusteringAlg.__name__))
        start = time.time()
        self.graph = nx.Graph()

        #dense cluster for each household
        self.clusterGroups('household', self.clusterDense, None)
        #cluster schools and workplaces with specified clustering alg
        if exemption != 'workplaces':
            self.clusterGroups('work', clusteringAlg, self.environment_masking['work'], [self.environment_degrees['work'], 0.5])
        if exemption != 'schools':
            self.clusterGroups('school', clusteringAlg, self.environment_masking['school'], [self.environment_degrees['school'], 0.5])
        stop_a = time.time()
        #self.record.print("Graph completed in {} seconds.".format((stop_a -
        self.isBuilt = True

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
                #edges[iPartition, jPartition] += 1
        #plt.imshow(np.array([row / np.linalg.norm(row) for row in contact_matrix]))
        #return edges

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

    def getR0(self):
        sim = self.sims[-1]
        herd_immunity = list.index(max(sim.I))
        return(self.population/sim.S([herd_immunity]))

    def clearSims(self):
        self.sims = []

    def matchTauWithR0(self,R0):
        assert (self.isBuilt, 'model must be built first')
        search_range = 0.05
        simBuffer = self.sims
        found = False
        while(not found):
            #new_tau = tau +(R0-tau)*(R0-)
            pass
        tau = 0.05

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

    def plotBars(self, partition, xlabels, SIRstatus):
        simCount = len(self.sims)
        partitionCount = len(partition)
        barGroupWidth = 0.8
        barWidth = barGroupWidth/simCount
        index = np.arange(partitionCount)

        offset = 0
        for sim in self.sims:
            title = sim[1]
            sim = sim[0]

            totals = []
            end_time = sim.t()[-1]
            for element in partition:
                totals.append(sum(status == SIRstatus for status in sim.get_statuses(element, end_time).values()) / len(element))
            #totals = sorted(totals)
            xCoor = [offset + x for x in list(range(len(totals)))]
            plt.bar(xCoor,totals, barWidth, label = title)
            offset = offset+barWidth
        plt.legend()
        plt.ylabel("Fraction of people with status {}".format(SIRstatus))
        plt.xlabel("Age groups of 5 years")
        plt.show()
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

