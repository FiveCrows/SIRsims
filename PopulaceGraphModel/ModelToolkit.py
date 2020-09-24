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
import math



class Partitioner:
    """
    Objects of this class can be used to split a list of people into disjoint sets
    """

    def __init__(self, attribute, enumerator, names=None):
        """
        :param attribute: string
        The attribute by which to partition must match one of the attributes in 'populace'

        :param enumerator: dict
        The enumerator should map each possible values for the given attribute to the index of a partition set

        :param names: list
        A list of names for plotting with partitioned sets
        """

        self.enumerator = enumerator
        self.attribute = attribute
        self.names = names
        self.attribute_values = dict.fromkeys(set(enumerator.values()))
        self.num_sets = (len(np.unique(list(enumerator.values()))))

    def partitionGroup(self, members, populace):
        """
        :param members: list
        An list of indexes for the peaple to partition
        :param populace:
        A dict associating people to a list of their attributes is required for applying the enumerator

        :return: dict

        """
        partitioned_members = {i: [] for i in range(self.num_sets)}
        for person in members:
            #determine the number for which group the person belongs in, depending on their attribute
            group = self.enumerator[populace[person][self.attribute]]
            #add person to  to dict in group
            partitioned_members[group].append(person)
        return partitioned_members

#was a thought, but I never used it
class memberedPartition:
    def __init__(self, members, populace, enumerator, attribute, names = None):
        super.__init__()
        self.partitioned_members = super.partitionGroup(members, populace)

class Environment:
    """
    Objects to the carry details of every home, workplace, and school
    """
    def __init__(self, index, members, type, preventions = None):
        """
        :param index: int
        an identifier
        :param members: list
        a list of people who attend the environment
        :param type: string
        either 'household', 'school', or 'workplace'
        :param preventions: dict
        keys should be 'household', 'school', or 'workplace'. Each should map to another dict,
        with keys for 'masking', and 'distancing', which should map to an int in range[0:1] that represents
        the prevelance of the prevention strategy in the environment
        """

        self.index = index
        self.members = members
        self.type = type
        self.preventions = None
        self.population = len(members)
       # self.distancing = distancing


class PartitionedEnvironment(Environment):
    """
    An environment that contains a partitioner,
    """
    def __init__(self, index, members, type, populace, contact_matrix, partitioner, preventions = None):
        """
        :param index: int
        an identifier
        :param members: list
        a list of people who attend the environment
        :param type: string
        either 'household', 'school', or 'workplace'
        :param preventions: dict
        keys should be 'household', 'school', or 'workplace'. Each should map to another dict,
        with keys for 'masking', and 'distancing', which should map to an int in range[0:1] that represents
        the prevelance of the prevention strategy in the environment
        :param populace: dict

        :param contact_matrix: 2d array

        :param partitioner: Partitioner
        :param preventions: dict
        """
        super().__init__(index,members, type, preventions)
        self.partitioner = partitioner
        self.contact_matrix = contact_matrix
        self.id_to_partition = dict.fromkeys(members)

        #self.total_matrix_contact = contact_matrix.sum()
        self.partition = partitioner.partitionGroup(members, populace)
        for set in self.partition:
            for person in self.partition[set]:
                self.id_to_partition[person] = (set)

    def returnReciprocatedCM(self):
        cm = self.contact_matrix
        dim = cm.shape
        rm = np.zeros(dim)
        set_sizes = [len(self.partition[i]) for i in self.partition]

        for i in range(dim[0]):
            for j in range(dim[1]):
                if set_sizes[i] != 0:
                    rm[i,j] = (cm[i,j]*set_sizes[i]+cm[j,i]*set_sizes[j])/(2*set_sizes[i])
        return rm
        def __str__(self):
            print("{}population: {}".type)


class TransmissionWeighter:
    def __init__(self, env_scalars, prevention_reductions, name ='default'):#, loc_masking):
        self.name = name
        self.global_weight = 1
        self.prevention_reductions = prevention_reductions
        self.env_scalars = env_scalars

        #self.loc_masking = loc_masking
        #self.age_scalars = age_scalars
    def getWeight(self, personA, personB, environment):
        weight = self.global_weight*self.env_scalars[environment.type]
        #including masks
        if environment.preventions != None:
            if random.random() < environment.preventions["masking"]:
                weight = weight * self.prevention_reductions["masking"]
                if random.random() < environment.preventions["masking"]**2:
                    weight = weight * self.prevention_reductions["masking"]
            #distancing weight reduction
            weight = weight*(1-(1-self.prevention_reductions["distancing"]) * environment.preventions["distancing"])
        return weight


class PopulaceGraph:
    "The class PopulaceGa"
    def __init__(self, partition = None, graph = None, populace = None, pops_by_category = None, categories = ['sp_hh_id', 'work_id', 'school_id', 'race', 'age'], slim = False):
        self.isBuilt = False
        #self.record = Record()
        self.sims = []
        self.contactMatrix = None
        self.total_weight = 0
        self.record = Record()
        self.total_edges = 0
        self.total_weight = 0
        self.environments_added = 0

        if graph == None:
            self.graph = nx.Graph()

        #load populace from file if necessary
        if populace == None:
        # for loading people objects from file
            with open("people_list_serialized.pkl", 'rb') as file:
                x = pickle.load(file)

            # return represented by dict of dicts
        #renames = {"sp_hh_id": "household", "work_id": "work", "school_id": "school"} maybe later...

        if slim == True:
            print("WARNING! slim = True, 90% of people are filtered out")
            self.populace = {}
            for key in x:
                if random.random()>0.9:
                    self.populace[key] = (vars(x[key]))
        else:
            self.populace = ({key: (vars(x[key])) for key in x})  # .transpose()
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

        #list households:

        #load contact_matrices and build environments
        with open("../ContactMatrices/Leon/ContactMatrixSchools.pkl", 'rb') as file:
            schoolCM = pickle.load(file)

        # env_name_alternate = {"household": "sp_hh_id", "work": "work_id", "school": "school_id"} outdated
        #adding households to environment list
        households = self.pops_by_category["sp_hh_id"]
        self.environments = {}
        for index in households:
            houseObject = Environment(index, households[index], "household", 0)
            self.environments[index] = (houseObject)

        #adding workplaces to environment list
        workplaces = self.pops_by_category["work_id"]
        with open("../ContactMatrices/Leon/ContactMatrixWorkplaces.pkl", 'rb') as file:
            work_matrices = pickle.load(file)

        if partition != None:
            self.hasPartition = True
            for index in workplaces:
                if index != None:
                    workplace = PartitionedEnvironment(index, workplaces[index], "workplace", self.populace, work_matrices[index], partition)
                    self.environments[index] = (workplace)
            schools = self.pops_by_category["school_id"]
            with open("../ContactMatrices/Leon/ContactMatrixSchools.pkl", 'rb') as file:
                school_matrices = pickle.load(file)
            for index in schools:
                if index != None:
                    school = PartitionedEnvironment(index, schools[index], "school", self.populace, school_matrices[index], partition )
                    self.environments[index] = (school)




    def build(self, weighter, preventions, env_degrees, alg = None):
        #none is default so old scripts can still run. self not defined in signature
        if alg == None:
            alg = self.clusterPartitionedStrogatz

        self.trans_weighter = weighter
        self.preventions = preventions
        self.environment_degrees = env_degrees
        #self.record.print('\n')
        #self.record.print("building populace into graphs with the {} clustering algorithm".format(clusteringAlg.__name__))
        #start = time.time()
        self.graph = nx.Graph()
        for index in self.environments:
            environment = self.environments[index]
            environment.preventions = preventions[environment.type]
            self.addEnvironment(environment, alg)
        self.isBuilt = True

    def addEdge(self, nodeA, nodeB, environment, weight_scalar = 1):
        weight = self.trans_weighter.getWeight(nodeA, nodeB, environment)*weight_scalar
        self.total_weight += weight
        self.total_edges += 1
        self.graph.add_edge(nodeA, nodeB, transmission_weight = weight)

    #merge environments, written for plotting and exploration
    def returnMultiEnvironment(self, env_indexes, partition):
        members = []
        for index in env_indexes:
            members.extend(self.environments[index].members)
        return PartitionedEnvironment(None, members, 'multiEnvironment', self.populace, None, partition)

    def clusterDense(self, environment, subgroup = None, weight_scalar = 1):
        if subgroup == None:
            members = environment.members
        else:
            members = subgroup
        type = environment.type
        member_count = len(members)
        #memberWeightScalar = np.sqrt(memberCount)
        for i in range(member_count):
            for j in range(i):
                self.addEdge(members[i], members[j], environment, weight_scalar)


    def addEnvironment(self, environment, alg):
        if environment.type == 'household':
            self.clusterDense(environment)
        else:
            # the graph is computed according to contact matrix of environment
            # self.clusterPartitionedStrogatz(environment, self.environment_degrees[environment.type])
            alg(environment, self.environment_degrees[environment.type])


    def clusterStrogatz(self, environment,  num_edges, weight_scalar = 1, subgroup = None, rewire_p = 0.2):
        if subgroup == None:
            members = environment.members
        else:
            members = subgroup

        #unpack params
        # if only one person, don't bother
        member_count = len(members)
        if member_count == 1:
            return

        local_k = math.floor(num_edges/member_count)*2
        remainder = num_edges - local_k*member_count/2
        if local_k >= member_count:
            self.clusterDense(environment, weight_scalar = weight_scalar)
            return

        for i in range(member_count):
            nodeA = members[i]
            for j in range(1, local_k // 2+1):
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = members[(i + random.choice(range(member_count - 1))) % member_count]
                else:
                    nodeB = members[(i + j) % member_count]
                self.addEdge(nodeA, nodeB, environment, weight_scalar)
        edgeList = self.genRandEdgeList(members, members, remainder)
        for edge in edgeList: self.addEdge(edge[0], edge[1], environment)


    def clusterBipartite(self, environment, members_A, members_B, edge_count, weight_scalar = 1, p_random = 0.2):
        #reorder groups by size
        A = min(members_A, members_B, key = len)
        if A == members_A:
            B = members_B
        else:
            B = members_A

        size_A = len(A)
        size_B = len(B)

        if len(members_A)*len(members_B) < edge_count:
            print("warning, not enough possible edges for cluterBipartite")

        #distance between edge groups
        separation = int(math.ceil(size_B/size_A))

        #size of edge groups and remaining edges
        k = edge_count//size_A
        remainder = edge_count%size_A
        p_random = max(0, p_random - remainder/edge_count)

        for i in range(size_A):
            begin_B_edges = (i * separation - k // 2)%size_B

            for j in range(k):
                if random.random()>p_random:
                    B_side = (begin_B_edges +j)%size_B
                    self.addEdge(A[i], B[B_side], environment, weight_scalar)
                else:
                    remainder +=1


        eList = self.genRandEdgeList(members_A, members_B, remainder)
        for edge in eList: self.addEdge(edge[0], edge[1],environment, weight_scalar)


    #for clusterRandGraph
    def genRandEdgeList(self, setA, setB, n_edges):
        if n_edges == 0:
            return []
        n_edges = int(n_edges)
        n_A = len(setA)
        n_B = len(setB)
        if setA == setB:
            pos_edges = n_A*(n_A-1)/2
            same_sets = True
        else:
            pos_edges = n_A*n_B
            same_sets = False

#        p_duplicate = n_edges/pos_edges
#        if p_duplicate< 0.001:
#            list = [(random.choice(setA),random.choice(setB)) for i in range(n_edges)]
#        else:
        edge_dict = {}
        while len(edge_dict)<n_edges:
            A, B = random.choice(setA), random.choice(setB)
            if A>B: edge_dict[A,B] = 1
            elif B>A: edge_dict[A,B] = 1
        list = edge_dict.keys()
        return list

    def clusterRandGraph(self, environment, avg_degree):
        print("**** Enter clusterRandGraph, created by G. Erlebacher")
        # Create graph according to makeGraph, developed by G. Erlebacher (in Julia)
        #G = Gordon()
        #G.makeGraph(N, index_range, cmm)
        # len(p_sets) = 16 age categories, dictionaries
        p_sets = environment.partition
        population = environment.population
        CM = environment.returnReciprocatedCM()
        print("CM= ", CM)  # single CM matrix

        assert isinstance(environment, PartitionedEnvironment), "must be a partitioned environment"
        #determine total edges needed for entire network. There are two connections per edge)
        #a list of the number of people in each partition set
        p_sizes = [len(p_sets[i]) for i in p_sets]
        num_sets = len(p_sets) # nb age bins
        #get total contact, keeping in mind the contact matrix elements are divided by num people in group
        total_contact = 0
        #for i in range(len(CM)):
        for i,row in enumerate(CM):
            #add total contacts for everyone in set i
            total_contact +=sum(np.array(row))*p_sizes[i]
        if avg_degree == None:
            total_edges = math.floor(total_contact) / 2
        else:
            total_edges = math.floor(avg_degree*population/ 2)

        #for every entry in the contact matrix
        edge_list = []
        for i,row in enumerate(CM):
            for j, cm in enumerate(row):
                #decide how many edges it implies
                if i == j:
                    n_edges = math.floor(cm/(total_contact*2))
                else:
                    n_edges = math.floor(cm/total_contact)
                if n_edges == 0:
                    continue
                edge_list = self.genRandEdgeList(p_sets[i], p_sets[j], n_edges)
                for edge in edge_list: self.addEdge(edge[0],edge[1],environment)

        #default_weight = total_contact/totalEdges

    def clusterPartitionedStrogatz(self, environment, avg_degree = None):
        self.clusterWithMatrix( environment, avg_degree, 'strogatz')

    def clusterPartitionedRandom(self, environment, avg_degree = None):
        self.clusterWithMatrix(environment, avg_degree, 'random')

    def clusterWithMatrix(self, environment, avg_degree, topology):
        #to clean up code just a little
        p_sets = environment.partition
        CM = environment.returnReciprocatedCM()

        assert isinstance(environment, PartitionedEnvironment), "must be a partitioned environment"
        #a list of the number of people in each partition set
        p_n = [len(p_sets[i]) for i in p_sets]
        num_sets = len(p_sets)
        #get total contact, keeping in mind the contact matrix elements are divided by num people in group
        total_contact = 0
        for i, row in enumerate(CM):
                total_contact += sum(row)*p_n[i]
        #default_weight = total_contact/totalEdges
        if avg_degree == None:
            avg_degree = total_contact/environment.population
        #print('by the sum of the CM, avg_degree should be : {}'.format(avg_degree ))
        #determine total edges needed for entire network. There are two connections per edge)
        total_edges = math.floor(avg_degree * environment.population/2)

        #for each number between two groups, don't iterate zeros
        for i in p_sets:
            for j in range(i, num_sets):
                if p_n[j] == 0:
                    continue
                #get the fraction of contact that should occur between sets i and j
                contactFraction = CM[i, j]*p_n[i]/(total_contact)
                if contactFraction == 0:
                    continue
                #make sure there are enough people to fit num_edges
                if i == j:
                    num_edges = int(total_edges * contactFraction)
                    max_edges = p_n[i] * (p_n[i]-1)
                else:
                    num_edges = int(total_edges*contactFraction*2)
                    max_edges = p_n[i] * p_n[j]
                if max_edges < num_edges:
                    num_edges = max_edges
                if num_edges == 0:
                    continue

                #if the expected number of edges cannot be added, compensate by scaling the weights up a bit
                residual_scalar = total_edges * contactFraction / num_edges
                #if residual_scalar>2 and sizeA>3:
                    #print("error in environment # {}, it's contacts count for i,j = {} is {}but there are only {} people in that set".format(environment.index, index_i, CM[index_i,index_j], len(environment.partitioned_members[index_i])))
                if topology == 'random':
                    edgeList = self.genRandEdgeList(p_sets[i], p_sets[j], num_edges)
                    for edge in edgeList:
                        self.addEdge(edge[0], edge[1], environment)
                else:
                    if i == j:
                        self.clusterStrogatz(environment, num_edges, weight_scalar =1, subgroup = p_sets[i])
                    else:
                        self.clusterBipartite(environment, p_sets[i], p_sets[j], num_edges,weight_scalar=1)







    #written for the clusterMatrixGuidedPreferentialAttachment function
    def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, environment):
        self.add_edge(nodeA, nodeB, environment)
        groupA = environment.id_to_partition[nodeA]
        groupB = environment.id_to_partition[nodeB]

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

    def clusterMatrixGuidedPreferentialAttachment(self, environment, avg_contacts, prob_rand):
        cumulative_weight = sum(sum(environment.contact_matrix))
        num_people = len(environment.members)
        total_pos_edges = num_people * (num_people - 1) / 2
        total_edges = num_people * avg_contacts
        random_edges = math.round(prob_rand * total_edges)
        remaining_edges = total_edges - random_edges
        vecM = np.matrix.flatten(environment.contact_matrix)
        num_partitions = len(vecM)
        partitionAttachments = {}


        # speed up, in case there aren't many duplicates likely anyways
        random_duplicate_rate = (random_edges - 1) / total_pos_edges
        # above, so much cancelled... hows that for some prob?
        if random_duplicate_rate > 0.01:
            rand_edges = random.choices(list(itertools.combinations(environment.members, 2)), k=random_edges)
            for edge in rand_edges:
                self.add_edge(edge[0], edge[1], environment)
        else:
            for i in range(random_edges):
                sel_A = random.choice(num_people)
                sel_B = (sel_A + random.choice(num_people - 1)) % num_people
                self.add_edge(environment.members[sel_A], environment.members[sel_B], environment)

        # now adding preferential attachment edges
        partition_dist = [sum(vecM[:i] for i in range(num_partitions))] / sum(vecM)
        # partition_dist projects the edge_partition to  [0,1], such that the space between elements is in proportion to
        # the elements contact
        for i in range(remaining_edges):
            # this selects a partition element using partition_dist
            # then, from vec back to row/col
            selector = random.random()
            raw_partition = list(filter(range(num_partitions),
                                        lambda i: partition_dist[i] < (selector) & partition_dist[i + 1] > (
                                        selector)))
            partition_A = raw_partition % environment.contact_matrix.shape[0]
            partition_B = raw_partition // environment.contact_matrix.shape[0]

            def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, id_to_partition, mask_p, weight):
                w = self.trans_weighter.genMaskScalar(mask_p) * weight
                self.graph.add_edge(nodeA, nodeB, transmission_weight=w)
                groupA = id_to_partition[nodeA]
                groupB = id_to_partition[nodeB]

                # grow secondary list
                # Adding B's friends to A's secondary
                for key in attachments[nodeA]["secondary"][nodeB]:
                    attachments[nodeA]["secondary"][key].extend(attachments[nodeB]["secondary"][key])
                # Adding A's friends to B's secondary
                for key in attachments[nodeB]["secondary"][nodeA]:
                    attachments[nodeB]["secondary"][key].extend(attachments[nodeA]["secondary"][key])

                # Adding B as secondary to A's friends
                for key in attachments[nodeA]:
                    pass
                # Adding A as secondary to B's friends

                # grow primary list,
                # adding B to A, A to B
                attachments[nodeA]["primary"][groupB].append(nodeB)
                attachments[nodeB]["primary"][groupA].append(nodeA)

                # Adding A's friends to B

                # Adding B to A's friends
                # Adding A to B's friends
                # try:
                # attachments[""]

    def simulate(self, gamma, tau, simAlg = EoN.fast_SIR, title = None, full_data = True):
        start = time.time()
        simResult = simAlg(self.graph, gamma, tau, rho=0.001, transmission_weight='transmission_weight', return_full_data=full_data)
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

    def returnContactMatrix(self, environment):
        graph = self.graph.subgraph(environment.members)
        partition = environment.partitioner
        contact_matrix = np.zeros([partition.num_sets, partition.num_sets])
        partition_sizes = [len(environment.partition[i]) for i in environment.partition]

        for i in graph.nodes():
            iPartition = environment.id_to_partition[i]
            contacts = graph[i]
            for j in contacts:
                jPartition = environment.id_to_partition[j]
                contact_matrix[iPartition, jPartition] += self.graph[i][j]['transmission_weight'] / partition_sizes[iPartition]
        
        
        # plt.imshow(np.array([row / np.linalg.norm(row) for row in contact_matrix]))
        return contact_matrix 


    def plotContactMatrix(self, p_env):
        '''
        This function plots the contact matrix for a partitioned environment
        :param p_env: must be a partitioned environment
        '''

        if p_env == None:
            p_env = self.returnMultiEnvironment()
        contact_matrix = self.returnContactMatrix(p_env)
        plt.imshow(contact_matrix)
        plt.title("Contact Matrix for members of {} # {}".format(p_env.type, p_env.index))
        labels = p_env.partitioner.labels
        if labels == None:
            labels = ["{}-{}".format(5 * i, (5 * (i + 1))-1) for i in range(15)]
        axisticks= list(range(15))
        plt.xticks(axisticks, labels, rotation= 'vertical')
        plt.yticks(axisticks, labels)
        plt.xlabel('Age Group')
        plt.ylabel('Age Group')
        plt.show()


    def plotNodeDegreeHistogram(self, environment = None, layout = 'bars', ax = None, normalized = True):
        """
        creates a histogram which displays the frequency of degrees for all nodes in the specified environment.
        :param environment: The environment to plot for. if not specified, a histogram for everyone in the model will be plotted
        :param layout: if 'lines', a line plot will be generated. otherwise a barplot will be used
        :param ax: if an pyplot axis is specified, the plot will be added to it. Otherwise, the plot will be shown
        :param normalized, when true the histogram will display the portion of total
        """

        if environment != None:
            people = environment.members
            graph = self.graph.subgraph(people)
            plt.title("Degree plot for members of {} # {}".format(environment.type, environment.index))
        else:
            graph = self.graph
            people = self.populace.keys()

        degreeCounts = [0] * 100
        for person in people:
            try:
                degree = len(graph[person])
            except:
                degree = 0
            degreeCounts[degree] += 1
        while degreeCounts[-1] == 0:
            degreeCounts.pop()
        if layout == 'lines':
            plt.plot(range(len(degreeCounts)), degreeCounts)
        else:
            plt.bar(range(len(degreeCounts)), degreeCounts)
        plt.ylabel("total people")
        plt.xlabel("degree")
        plt.show()
        plt.savefig("./simResults/{}/".format(self.record.stamp))


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

    #If a partitionedEnvironment is specified, the partition of the environment is applied, otherwise, a partition must be passed
    def plotBars(self, environment = None, SIRstatus = 'R', normalized = False):
        """
        Will show a bar chart that details the final status of each partition set in the environment, at the end of the simulation
        :param environment: must be a partitioned environment
        :param SIRstatus: should be 'S', 'I', or 'R'; is the status bars will represent
        :param normalized: whether to plot each bar as a fraction or the number of people with the given status

        """
        partition = environment.partitioner
        if isinstance(environment, PartitionedEnvironment):
            partitioned_people = environment.partition
            partition = environment.partitioner

        simCount = len(self.sims)
        partitionCount = partition.num_sets
        barGroupWidth = 0.8
        barWidth = barGroupWidth/simCount
        index = np.arange(partitionCount)

        offset = 0
        for sim in self.sims:
            title = sim[1]
            sim = sim[0]

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
        plt.savefig("./simResults/{}/evasionChart".format(self.record.stamp))

    def getR0(self):
        sim = self.sims[-1]
        herd_immunity = list.index(max(sim.I))
        return(self.population/sim.S([herd_immunity]))

    def reset(self):
        self.sims = []
        self.graph = nx.Graph()
        self.total_weight = 0
        self.total_edges = 0
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

#----------------------------------------------------------------------
class Gordon:
    def __init__(self):
        pass

    def reciprocity(self, cm, N):
        # The simplest approach to symmetrization is Method 1 (M1) in paper by Arregui
        cmm = np.zeros([4,4])
        for i in range(4):
            for j in range(4):
                if N[i] == 0:
                    cmm[i,j] = 0
                else:
                    cmm[i,j] = 1/(2 * N[i])  * (cm[i,j]*N[i] + cm[j,i]*N[j])
    
                if N[j] == 0:
                    cmm[j,i] = 0
                else:
                    cmm[j,i] = 1/(2 * N[j])  * (cm[j,i]*N[j] + cm[i,j]*N[i])
        return cmm
    
    #---------------------------------------------
    # This is GE's algorithm, a copy of what I implemented in Julia. 
    # We need to try both approaches for our paper. 
    def makeGraph(self, N, index_range, cmm):
        # N: array of age category sizes
        # index_range: lo:hi tuple 
        # cmm: contact matrix with the property: cmm[i,j]*N[i] = cmm[j,i]*N[j]
        # Output: a list of edges to feed into a graph
    
        edge_list = []
        Nv = sum(N)
        if Nv < 25: return edge_list # <<<<< All to all connection below Nv = 25. Not done yet.
    
        lo, hi = index_range
        # Assign age groups to the nodes. Randomness not important
        # These are also the node numbers for each category, sorted
        age_bins = [np.repeat([i], N[i]) for i in range(lo,hi)]
    
        # Efficiently store cummulative sums for age brackets
        cum_N = np.append([0], np.cumsum(N))
    
        ddict = {}
        total_edges = 0
    
        print("lo,hi= ", lo, hi)
        for i in range(lo,hi):
            for j in range(lo,i+1):
                #print("lo,i= ", lo, i)
                ddict = {}
                Nij = int(N[i] * cmm[i,j])
                print("i,j= ", i, j, ",    Nij= ", Nij)
    
                if Nij == 0:
                    continue 
    
                total_edges += Nij
                # List of nodes in both graphs for age brackets i and j
                Vi = list(range(cum_N[i], cum_N[i + 1]))  # Check limits
                Vj = list(range(cum_N[j], cum_N[j+1]))  # Check limits

                # Treat the case when the number of edges dictated by the
                # contact matrices is greater than the number of available edges
                # The connectivity is then cmoplete
                if i == j:
                    lg = len(Vi)
                    nbe = lg*(lg-1) // 2
                else:
                    nbe = len(Vi)*len(Vj)
                    if nbe == 0:
                        continue

                if Vi == Vj and Nij > nbe:
                    Nij = nbe
    
                count = 0
    
                while True:
                    # p ~ Vi, q ~ Vj
                    # no self-edges
                    # only reallocate when necessary (that would provide speedup)
                    # allocate 1000 at t time
                    #p = getRand(Vi, 1) # I could use memoization
                    #q = getRand(Vi, 1) # I could use memoization
    
                    #p = rand(Vi, 1)[]
                    #q = rand(Vj, 1)[]
                    p = random.choice(Vi)
                    q = random.choice(Vj)

                    # multiple edges between p,q not allowed
                    # Dictionaries only store an edge once
                    if p == q: continue
                    if p > q:
                        ddict[p, q] = 1
                    elif q > p:
                        ddict[q, p] = 1

                    # stop when desired number of edges is reached
                    lg = len(ddict)
                    if lg == Nij: break 
    
                for k in ddict.keys():
                    s, d = k
                    edge_list.append((s,d))
    
        print("total_edges: ", total_edges)
        print("size of edge_list: ", len(edge_list))
        return edge_list
    #------------------------------------------------------------------
