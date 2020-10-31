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
from scipy.interpolate import interp1d
from scipy.stats import bernoulli


class Partitioner:
    """
    Objects of this class can be used to split a list of people into disjoint sets
    """

    def __init__(self, attribute, enumerator, labels=None):
        """
        :param attribute: string
        The attribute by which to partition must match one of the attributes in 'populace'

        :param enumerator: dict
        The enumerator should map each possible values for the given attribute to the index of a partition set

        :param labels: list
        A list of names for plotting with partitioned sets
        """

        self.enumerator = enumerator
        self.attribute = attribute
        self.labels = labels
        self.attribute_values = dict.fromkeys(set(enumerator.values()))
        self.num_sets = (len(np.unique(list(enumerator.values()))))

    def partitionGroup(self, members, populace):
        """
        :param members: list
        A list of indexes for the peaple to partition, if 'All' will use all members in populace
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


class Environment:
    """
    Objects to the carry details for every home
    """

    def __init__(self, index, members, quality):
        """
        :param index: int
        an identifier
        :param members: list
        a list of people who attend the environment
        :param type: string
        either 'household', 'school', or 'workplace'
        """

        self.index = index
        self.members = members
        self.quality = quality
        self.population = len(members)
        self.num_mask_types = 1
        # self.distancing = distancing
        self.total_weight = 0
        self.edges = []
        #booleans to keep track of build process
        self.hasEdges = False
        self.isWeighted = False
        #creates a dict linking each member of the environment with each prevention

    def drawPreventions(self, prevalences, populace):
        """
        picks masks and distancing parameters
        :param prevalences: dict
        the prevalences for each prevention, with keys for environment type, to prevention type to prevalence
        :param populace: dict
        the populace dict is needed to know which mask everybody is using, if necessary

        :return:
        """

        prevalences = prevalences[self.quality]
        #assign distancers
        num_distancers = int(self.population * prevalences["distancing"])
        distance_status = [1] * num_distancers + [0] * (self.population - num_distancers)
        random.shuffle(distance_status)

        num_masks = int(self.population * prevalences["masking"])

        mask_status = [1] * num_masks + [0] * (self.population - num_masks)

        random.shuffle(mask_status)
        if num_masks != 1:
            # originally, 1 to represent does wear mask, this is replaced by an int to represent the type of mask worn
            for index in range(len(mask_status)):
                if mask_status[index] == 1:
                    mask_status[index] = self.populace.members[index]["mask_type"]

        self.mask_status = dict(zip(self.members, mask_status))
        self.distance_status = dict(zip(self.members, distance_status))


    def reweight(self, netBuilder, newPreventions = None):
        """
        Rechooses the weights on each edge with, presumably, a distinct weighter or preventions
        :param weighter: TransmissionWeighter object
        will be used to determine the graphs weights
        :param preventions: dict
        should associate each environment type to another dict, which associates each prevention to a prevalence
        :return: None
        """

        #update new prevention strategies on each environment
        #recalculate weight for each edge
        for edge in self.edges:
            new_weight = netBuilder.getWeight(edge[0], edge[1], self)
            edge[2] = new_weight


    def addEdge(self, nodeA, nodeB, weight):
        '''
        fThis helper function  not only makes it easier to track
        variables like the total weight and edges for the whole graph, it can be useful for debugging
        :param nodeA: int
         Index of the node for one side of the edge
        :param nodeB: int
        Index of the node for the other side
        :param weight: double
         the weight for the edge
        '''

        self.total_weight += weight
        self.edges.append([nodeA, nodeB, weight])

    def network(self, netBuilder):
        netBuilder.buildDenseNet(self)

    def clearNet(self):
        self.edges = []
        self.total_weight = 0


class StructuredEnvironment(Environment):
    """
    These environments are extended with a contact matrix and partition
    """

    def __init__(self, index, members, quality, populace, contact_matrix, partitioner, preventions = None):
        """
        :param index: int
        to index the specific environment
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
        for creating a partition
        """
        super().__init__(index,members, quality)
        self.partitioner = partitioner
        self.contact_matrix = contact_matrix
        self.id_to_partition = dict.fromkeys(members)

        #self.total_matrix_contact = contact_matrix.sum()
        self.partition = partitioner.partitionGroup(members, populace)
        for set in self.partition:
            for person in self.partition[set]:
                self.id_to_partition[person] = (set)

    def returnReciprocatedCM(self):
        '''
        :return: this function  averages to returs a modified version of the contact matrix where
        CM_[i,j]*N[i]= CM_[j,i]*N[j]
        '''

        cm = self.contact_matrix
        dim = cm.shape
        rm = np.zeros(dim)
        set_sizes = [len(self.partition[i]) for i in self.partition]

        for i in range(dim[0]):
            for j in range(dim[1]):
                if set_sizes[i] != 0:
                    rm[i,j] = (cm[i,j]*set_sizes[i]+cm[j,i]*set_sizes[j])/(2*set_sizes[i])
        return rm

    def network(self, netBuilder):
        netBuilder.buildStructuredNet(self)


class NetBuilder:
    """
    This class is written to hold network building algorithms
    The Netbuilder can be called to create a densenets for Environment
    and Random nets for partitioned Environments.
    """

    def __init__(self, env_type_scalars, prev_efficacies, cv_dict = {}, avg_contacts = None):
        """
        :param env_type_scalars: dict
        each environment type must map to a float. is for scaling weights

        :param prev_efficacies: dict
        must map prevention names, currently either 'masking' or 'distancing'. Can map to floats, but
        masking has the option of mapping to a list. In that case, the script will attempt to
        determine,  for the environment, which masks each person should wear, and  pick the appropriate
        scalar from the list. index [0] will represent no mask, 1 up will be mask types

        :param avg_contacts:
        if specified, the number of edges picked for an environment will be chosen to meet avg_contacts

        :param cv_dict: dict
        the cv dict allows the user to specify values for keys "weight", "contact", and "mask_eff",
        which will be used as the coefficient of variation for applying noise to these parrameters,
        noise to the weights, the number of contacts in structured environments, and the efficacy of masks
        """

        self.global_weight = 1
        self.prev_efficacies = prev_efficacies
        self.env_scalars = env_type_scalars
        self.cv_dict = cv_dict
        self.avg_contacts = avg_contacts

    #def list
    def addEdge(self, nodeA, nodeB, environment):
        """
        just gets the weight and calls it into the environment
        :return:
        """

        weight = self.getWeight(nodeA, nodeB, environment)
        environment.addEdge(nodeA, nodeB, weight)

    def buildDenseNet(self, environment, subgroup=None, weight_scalar = 1):
        """
        This function will add every edge possible for the environment. Thats n*(n-1)/2 edges
        :param environment: Environment
        The environment to add edges to
        :param subgroup: list
        may be used if one intends to add edges for only members of the environments subgroup
        :param weight_scalar: double
         may be used if one wants the weights scaled larger/smaller than normal

        :return edge_list: ebunch
        returns a list of weighted edges in form (nodeA, nobeB, weight)
        """

        if subgroup == None:
            members = environment.members
        else:
            members = subgroup
        #quality = environment.quality
        member_count = len(members)

        for i in range(member_count):
            for j in range(i):
                nodeA, nodeB = members[i], members[j]
                weight = self.getWeight(nodeA, nodeB, environment)
                environment.addEdge(nodeA, nodeB, weight)


    def genRandEdgeList(self, setA, setB, n_edges):
        if n_edges == 0:
            return []
        n_edges = int(n_edges)
        n_A = len(setA)
        n_B = len(setB)
        if setA == setB:
            pos_edges = n_A * (n_A - 1) / 2
            same_sets = True
        else:
            pos_edges = n_A * n_B
            same_sets = False

        #        p_duplicate = n_edges/pos_edges
        #        if p_duplicate< 0.001:
        #            list = [(random.choice(setA),random.choice(setB)) for i in range(n_edges)]
        #        else:
        edge_dict = {}
        while len(edge_dict) < n_edges:
            A, B = random.choice(setA), random.choice(setB)
            if A > B:
                edge_dict[A, B] = 1
            elif B > A:
                edge_dict[A, B] = 1
        list = edge_dict.keys()
        return list

    # for clusterRandGraph
    def buildBipartiteNet(self, environment, members_A, members_B, edge_count, weight_scalar = 1, p_random = 0.2):
        """
        cluster bipartite is for linking edges between two disjoint sets of people in the given environment
        :param environment: environment Object
        :param members_A: list
        a list of people
        :param members_B: list
        another list of people
        :param edge_count: int
        the number of edges to add to the environment
        :param weight_scalar: int
        :param p_random: int
        the rate at which random edges need to be added
        :return:
        """
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
                    nodeA, nodeB =A[i], B[(begin_B_edges +j)%size_B]
                    weight = self.getWeight(nodeA, nodeB, environment)
                    environment.addEdge(nodeA,nodeB,weight)
                else:
                    remainder +=1


        eList = self.genRandEdgeList(members_A, members_B, remainder)
        for edge in eList:
            weight = self.getWeight(edge[0], edge[1], environment)
            environment.addEdge(edge[0], edge[1], weight)


    def buildStructuredNet(self, environment, avg_degree = None):
        """
        For a partitioned environment, its reciprocated contact matrix
        to  be used to determine the number of edges needed between each pair of partition sets to match as closely as possible.
        In cases where the contact matrix expects more edges than are possible given two sets, the algorithm will add just the max possible
        edges. The edges are  then placed randomly, between members of their assigned sets

        :param environment: a PartitionedEnvironment
        the environment to add edges for

        :param avg_degree: int or double
         if avg_degree is not None, then the contact matrix should scaled such that the average degree

        :param topology: string
        can be either 'random', or 'strogatz'

        :return:
        """

        #to clean up code just a little
        p_sets = environment.partition
        CM = environment.returnReciprocatedCM()

        #add gaussian noise to contact matrix values
        if "contact" in self.cv_dict: 
            CM = CM*np.random.normal(1, self.cv_dict["contact"], CM.shape)
            CM[np.where(CM < 0.)] = 0.

        assert isinstance(environment, StructuredEnvironment), "must be a partitioned environment"
        #a list of the number of people in each partition set
        p_n      = [len(p_sets[i]) for i in p_sets]
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

                #not worth the concern atm tbh
                #to  compensate by scaling the weights up a bit
                #residual_scalar = total_edges * contactFraction / num_edges
                #if residual_scalar>2 and sizeA>3:
                    #print("error in environment # {}, it's contacts count for i,j = {} is {}but there are only {} people in that set".format(environment.index, index_i, CM[index_i,index_j], len(environment.partitioned_members[index_i])))

                edgeList = self.genRandEdgeList(p_sets[i], p_sets[j], num_edges)
                for edge in edgeList:
                    self.addEdge(edge[0], edge[1], environment)


    def setEnvScalars(self, env_scalars):
        self.env_scalars = env_scalars

    def setPreventions(self, preventions):
        self.preventions = preventions

    def setPreventionReductions(self, prevention_reductions):
        self.prevention_reductions = prevention_reductions

    def getWeight(self, personA, personB, environment):
        """
        Uses the environments type and preventions to deternmine weight
        :param personA: int

        Uses the environments type and preventions to deternmine weight
        :param personA: int
        a persons index, which should be a member of the environment
        :param personB: int
        :param environment: environment
         the shared environment of two nodes for the weight
        :return:
        """
        weight = self.global_weight*self.env_scalars[environment.quality]
        mask_eff = self.prev_efficacies["masking"]
        #factor with masks and distancing
        #if there are different masks in use, a different method is required

        #print("GE: env mask_types: ", environment.num_mask_types)  # 1 (instead of 3)
        if environment.num_mask_types == 1:
            n_masks = (environment.mask_status[personA] + environment.mask_status[personB])
            #so it works with reductions as either a single value, for one mask type in the model, or multiple vals, for multiple mask types
            # n_masks is 0,1, or 2. For each mask worn, weight is scaled down by reduction
            weight = weight * (1 - mask_eff) ** n_masks

        # handle situations where the model is set up with multiple different sorts of masks in use
        else:
            # This check for each person, is extremely inefficient!
            if len(environment.num_mask_types) != len(mask_eff["masking"]):
                print("warning: number of mask types does not match list size for reduction factors")
            #reduction factors for the type of mask person A and B wear
            redA = mask_eff[environment.mask_status[personA]]  # <<< ERROR?
            redB = mask_eff["masking"][environment.mask_status[personB]]

            # Apply spread to mask effectiveness if requested
            if "mask_eff" in self.cv_dict: 
                redA = redA*self.cv_dict["mask_eff"] 
                redA[np.where[redA < 0.]] = 0.
                redA[np.where[redA > 1.]] = 1.
                redB = redB*self.cv_dict["mask_eff"]
                redB[np.where[redB < 0.]] = 0.
                redB[np.where[redB > 1.]] = 1.

            # Next line assumes that two distancers don't double distance, but at least one distancer 
            # is needed to be distanced, will be 1 or 0
            weight = weight*(1-redA)*(1-redB)        

        isDistanced = int(bool(environment.distance_status[personA]) or bool(environment.distance_status[personB]))
        # only applies when isDistanced is 1
        weight = weight*(1-self.prev_efficacies["distancing"])**isDistanced

        # Add noise to the weight, if requested
        if "weight" in self.cv_dict: 
            scal = np.random.normal(1, self.cv_dict["weight"])
            if scal < 0.: scal = 0.
            if scal > 1.: scal = 1.
            weight = weight * scal
    
        # apply spread to mask effectiveness if requested

        """
        # This was in uncommented in ModelingToolkit2.py (2020-10-29) and generated an error. 
        # redA not defined. 
        if "mask_eff" in self.cv_dict: 
            redA = redA * self.cv_dict["mask_eff"]
            redB = redB * self.cv_dict["mask_eff"]
        """

        return weight

#A work in progress
class StrogatzNetBuilder(NetBuilder):

    def netStrogatz(self, environment,  num_edges, weight_scalar = 1, subgroup = None, rewire_p = 0.1):
        """
         netStrogatz creates a strogatz net in the given environment

         :param environment: Environment object,
        where to add edges
         :param num_edges: int
         the number of edges to add to the environment
         :param weight_scalar: int
          optional in case a larger or smaller weight is desired than transmission_weighter returns by default
         :param subgroup: list
         optional in case only edges for select members of the environment are wanted
         :param rewire_p:
         the portion of edges to be included into the net by random
         :return:
        """

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
            self.buildDenseNet(environment, weight_scalar = weight_scalar)
            return

        for i in range(member_count):
            nodeA = members[i]
            for j in range(1, local_k // 2+1):
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = members[(i + random.choice(range(member_count - 1))) % member_count]
                else:
                    nodeB = members[(i + j) % member_count]
                weight = self.getWeight(nodeA, nodeB, environment)
                environment.addEdge(nodeA, nodeB, weight)
        edgeList = self.genRandEdgeList(members, members, remainder)

        for edge in edgeList:
            weight = self.getWeight(nodeA, nodeB, environment)
            environment.addEdge(edge[0], edge[1], weight)

    def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, environment):
        """
        not finished yet.
        #written for the clusterMatrixGuidedPreferentialAttachment function

        :param nodeA:
        :param nodeB:
        :param attachments:
        :param environment:
        :return:
        """

        self.addEdge(nodeA, nodeB, environment)
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
        if random_duplicate_rate > 0.01:
            rand_edges = random.choices(list(itertools.combinations(environment.members, 2)), k=random_edges)
            for edge in rand_edges:
                self.addEdge(edge[0], edge[1], environment)
        else:
            for i in range(random_edges):
                sel_A = random.choice(num_people)
                sel_B = (sel_A + random.choice(num_people - 1)) % num_people
                self.addEdge(environment.members[sel_A], environment.members[sel_B], environment)

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
                self.graph.addEdge(nodeA, nodeB, transmission_weight = w)
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


class PopulaceGraph:
    """
    A list of people, environments, and functions, for tracking a weighted graph to represent contacts between members of the populace
    """

    def __init__(self, partitioner, prevention_prevalences = None, attributes = ['sp_hh_id', 'work_id', 'school_id', 'race', 'age'], slim = False, timestamp=None):
        """        
        :param partition: Partitioner
        needed to build schools and workplaces into partitioned environments
        :param attributes:
        names for the characteristics to load for each person
        :param slim: bool
        if set to True, will filter 90% of people. Mainly for debugging
        :param prevention_prevalences: dict
        keys should be 'household', 'school', or 'workplace'. Each should map to another dict,
        with keys for 'masking', and 'distancing', which should map to an int in range[0:1] that represents
        the prevelance of people practicing the prevention strategy in the environment
        """

        self.isBuilt = False
        #self.record = Record()
        self.sims = []
        self.contactMatrix = None
        self.total_weight = 0
        self.total_edges = 0
        self.total_weight = 0
        self.environments_added = 0
        self.initial_recovered = []
        self.initial_vaccinated = [] #None  # same as initial_recovered
        self.initial_infected   = [] #None
        self.social_distancing_reduction = None
        self.mask_reduction = None
        self.graph = nx.Graph()

        if timestamp == None:
            self.timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        else:
            self.timestamp = timestamp

        self.basedir = "./ge_simResults/{}".format(self.timestamp)
        print("basedir= ", self.basedir)
        self.record = Record(self.basedir)
        self.preventions = None
        self.prevention_reductions = None


        if prevention_prevalences == None:
            self.prevention_prevalences = {"household": {"masking": 0, "distancing": 0},
                                           "school": {"masking": 0, "distancing": 0},
                                           "workplace": {"masking": 0, "distancing": 0}}
        else:
            self.prevention_prevalences = prevention_prevalences

        # for loading people objects from file
        with open("people_list_serialized.pkl", 'rb') as file:
            rawPopulace = pickle.load(file)


        #reprocess rawPopulace into a list of lists of attributes
        if slim == True:
            print("WARNING! slim = True, 90% of people are filtered out")
            self.populace = []
            for key in rawPopulace:
                if random.random()>0.9:
                    self.populace.append(vars(rawPopulace[key]))
        else:
            self.populace = [(vars(rawPopulace[key])) for key in rawPopulace]  # .transpose()
        self.population = len(self.populace)
        #print("self.populace: ", self.populace); quit()
        print("self.population: ", self.population)


    # To sort people into a dict of categories. 
    # For example, if one wants the indexes of all people with the some school id, they could do
    #    pops_by_category['school_id'][someInt]
    # takes a dict of dicts to represent populace and returns a list of dicts of 
    #    lists to represent groups of people with the same attributes
    #  Give an example fo person[category], and an example of category
        pops_by_category = {category: {} for category in attributes}

        for index in range(len(self.populace)):
            person = self.populace[index]
            for category in attributes:
                try:
                    pops_by_category[category][person[category]].append(index)
                except:
                    pops_by_category[category][person[category]] = [index]

        #**************************88
        pops_by_category["age_groups"] = {}
        # all ages 0-90, 93, 94
        #print("age.keys(): ", sorted(list(pops_by_category["age"].keys())));
        count = 0
        for k,v in pops_by_category["age"].items():
            count = count + len(v)
        #print("__init__: total number of people: ", count) # same as self.population above. GOOD

        for bracket in range(0,20):
            pops_by_category["age_groups"][bracket] = []
            for i in range(0,5):
                try:   # easier than conditionals. I divided all ages into groups of 5
                    pops_by_category["age_groups"][bracket].extend(pops_by_category["age"][5*bracket+i])
                except:
                    continue

        # count total nb of nodes
        count=0
        for bracket in range(0,20):
            count = count + len(pops_by_category["age_groups"][bracket])
        print("PopulaceGraph::__init__: Nb of people in all age brackets: ", count)

        self.pops_by_category = pops_by_category

        # env_name_alternate = {"household": "sp_hh_id", "work": "work_id", "school": "school_id"} outdated
        #adding households to environment list
        self.environments = {}
        self.setup_households()
        self.setup_workplaces(partitioner)  # Partitin not defined BUG GE
        self.setup_schools(partitioner)

        # pick who masks and distances, in each environment
        for index in self.environments:
            self.environments[index].drawPreventions(self.prevention_prevalences, self.populace)
        #**************************88

        self.pops_by_category = pops_by_category

        print("test printEnvironments")
        self.printEnvironments()

        # Must be called last
        self.resetVaccinated_Infected()

    #-------------------------------------------------
    def printEnvironments(self):
        keys = list(self.environments.keys())
        envs = set()
        for k in keys:
            envs.add(self.environments[k].quality)

        # Environment type can be "household", "workplace", "school"
        #print("envs= ", envs)
        # attributers of environments[keys[1]]: "index', 'members', 'population', 'preventions', 'type'"
        #print(dir(self.environments[keys[1]]))
        for k in range(25000,25200):
            print("-----------------")
            print(self.environments[keys[k]].members)  # list of one element [12]. Meaning?
            print(self.environments[keys[k]].population)  # 1  (nb of members)
            print(self.environments[keys[k]].preventions)  # None (or a list?
            print(self.environments[keys[k]].quality)  # list of one element [12]. Meaning?

    #-------------------------------------------------
    def resetVaccinated_Infected(self):
        # By default nobody in the population is recovered. 
        # Vaccination is modeled by setting a person's status to recovered

        # Reset to not vaccinate anybody anywhere 
        self.set_nbTopWorkplacesToVaccinate(0, 0.)
        self.set_nbTopSchoolsToVaccinate(0, 0.)

        # Rank locations. Must be called after setting nb places to vaccinate
        print("** resetVaccinated_Infected")
        self.rank_workplaces()
        self.rank_schools()

        self.initial_vaccinated = []
        self.initial_infected   = []
        self.nb_top_workplaces_vaccinated = 0
        self.nb_top_schools_vaccinated = 0
        self.perc_people_vaccinated_in_workplaces = 0.0
        self.perc_populace_vaccinated = 0.0
        self.perc_people_vaccinated_in_schools = 0.0
        self.perc_workplace_vaccinated = 0.0  # perc vaccinated in the workplace
        self.perc_school_vaccinated = 0.0  # perc vaccinated in the schools
        self.work_population = 1
        self.school_population = 1
        self.workplace_population_vaccinated = 0,
        self.school_population_vaccinated = 0,
        self.initial_vaccinated_population = len(self.initial_vaccinated),

    #----------------------------------------------------------
    def createVaccinationDict(self):
        vacc_dict = {
            'nb_top_workplaces_vaccinated': self.nb_top_workplaces_vaccinated,
            'nb_top_schools_vaccinated': self.nb_top_schools_vaccinated,
            'perc_people_vaccinated_in_workplaces': self.perc_people_vaccinated_in_workplaces,
            'perc_populace_vaccinated': self.perc_populace_vaccinated,
            'perc_people_vaccinated_in_schools': self.perc_people_vaccinated_in_schools,
            'perc_workplace_vaccinated': self.perc_workplace_vaccinated,    # perc vaccinated in the workplace
            'perc_school_vaccinated': self.perc_school_vaccinated, # perc vaccinated in the schools
            'work_population': self.work_population,
            'school_population': self.school_population,
            'nb_schools': self.nb_schools,
            'nb_workplaces': self.nb_workplaces,
            'workplace_population_vaccinated': self.workplace_population_vaccinated,
            'school_population_vaccinated': self.school_population_vaccinated,
            'initial_vaccinated_population': self.initial_vaccinated_population,
        }
        return vacc_dict
    #-------------------------------
    def zeroWeights(self, env):
        # GE: How to turn off a school without reconstructing the school graph? 
        # NOT USED
        # Run through the edges and set the weight to zero
        start_time = time.time()
        G = self.graph
        for e in G.edges():
            environment = self.environments[self.graph.adj[e[0]][e[1]]['environment']]
            print("env, envi= ", env, environment.quality)
            if env == environment.quality:
                self.graph.adj[e[0]][e[1]]['transmission_weight'] = 0
        print("zeroWeights[%s]: %f sec" % (env, time.time() - start_time))
    #---------------------------------
    def printWeights(self):
        G = self.graph
        for e in G.edges():
            w = G[e[0]][e[1]]['transmission_weight']
            env = self.environments[G[e[0]][e[1]]['environment']]
            print("weight(e): ", e, " => ", w, env.quality)
    #---------------------------------
    def envEdges(self):
        # Added by Gordon Erlebacher, class PopulaceGraph
        # Construct a dictionary for each environment, of its edges. This will be used
        # to assign edge properties per environment. 
        env_edges= {}
        for ix in self.environments:
            env_edges[ix] = []   # list of edges

        for edge in self.graph.edges():
            #get environment
            env = self.environments[self.graph.adj[edge[0]][edge[1]]['environment']]
            # env is an environment object
            env_edges[env.index].append((edge[0], edge[1]))  ### ERROR

        # attach edge list to the environment
        for ix in self.environments:
            env = self.environments[ix]
            env.edges = env_edges[ix]

    #-------------------------------
    def setup_households(self):
        households = self.pops_by_category["sp_hh_id"]

        for index in households:
            houseObject              = Environment(index, households[index], "household")
            self.environments[index] = (houseObject)

    #-----------------
    def setup_workplaces(self, partitioner):
        self.workplaces = self.pops_by_category["work_id"]
        with open("../ContactMatrices/Leon/ContactMatrixWorkplaces.pkl", 'rb') as file:
            work_matrices = pickle.load(file)

        for index in self.workplaces:
            if index == None: continue
            #workplace = PartitionedEnvironment(index, workplaces[index], "workplace",   # Old code. Name change
            workplace = StructuredEnvironment(index, self.workplaces[index], "workplace", 
                                               self.populace, work_matrices[index], partitioner)
            self.environments[index] = (workplace)
        return

    #-----------------
    def setup_schools(self, partitioner):
        self.schools = self.pops_by_category["school_id"]
        with open("../ContactMatrices/Leon/ContactMatrixSchools.pkl", 'rb') as file:
            school_matrices = pickle.load(file)

        if partitioner == None: return
        self.hasPartition = True

        for index in self.schools:
            if index == None: continue
            #school = PartitionedEnvironment(index, schools[index], "school", self.populace,   # Old code, name change
            school = StructuredEnvironment(index, self.schools[index], "school", self.populace, 
                                            school_matrices[index], partitioner)
            self.environments[index] = (school)

    #-----------------
    def set_nbTopWorkplacesToVaccinate(self, nb, perc):
        self.nb_top_workplaces_vaccinated = nb
        self.perc_people_vaccinated_in_workplaces = perc
        print("*workplaces to vaccinate: ", self.nb_top_workplaces_vaccinated)
        print("*perc to vaccinate in workplaces: ", self.perc_people_vaccinated_in_workplaces)

    #---------------------
    def set_nbTopSchoolsToVaccinate(self, nb, perc):
        self.nb_top_schools_vaccinated = nb
        self.perc_people_vaccinated_in_schools = perc

    #------------------
    def rank_schools(self):
        # produces list of pairs (school is, list of people is)
        # replace the list of people ids by its length
        ordered_schools = sorted(self.schools.items(), key=lambda x: len(x[1]), reverse=True)
        ordered_schools = map(lambda x: [x[0], len(x[1])], ordered_schools)
        ordered_schools = list(ordered_schools)

        self.ordered_school_ids = [o[0] for o in ordered_schools[:]]
        self.ordered_school_pop = [o[1] for o in ordered_schools[:]]

        # Cumulative sum of school lengths
        # Sum from 1 since 0th index is a school with 200,000 students. Can't be right. 
        self.cum_sum_school_pop = np.cumsum(self.ordered_school_pop[1:])
        self.school_population = self.cum_sum_school_pop[-1]
        self.nb_schools = len(self.cum_sum_school_pop)
        self.largest_schools_vaccinated = self.ordered_school_ids[1:self.nb_top_schools_vaccinated]
        
        print("******* ENTER rank_schools *********")
        print("school_pop: ", self.ordered_school_pop[0:10])
        print("school_ids: ", self.ordered_school_ids[0:10])
        print("cum_sum, top 10: ", self.cum_sum_school_pop[0:10])
        print("rank_schools: self.nb_schools= ", self.nb_schools)
        print("* total school population: ", self.school_population)
        print("* total school population to vaccinate: ", self.cum_sum_school_pop[self.nb_top_schools_vaccinated])
        print("* school_id[0:10]: ", self.ordered_school_ids[0:10])
        print("******* EXIT rank_schools *********")

    #----------------------------------
    def rank_workplaces(self):
        # produces list of pairs (workplace is, list of people is)
        # replace the list of people ids by its length

        ordered_workplaces = sorted(self.workplaces.items(), key=lambda x: len(x[1]), reverse=True)
        ordered_workplaces = map(lambda x: [x[0], len(x[1])], ordered_workplaces)
        ordered_workplaces = list(ordered_workplaces)

        self.ordered_work_ids = [o[0] for o in ordered_workplaces]
        self.ordered_work_pop = [o[1] for o in ordered_workplaces]

        # Cumulative sum of business lengths
        # Sum from 1 since 0th index is a workplace with 120,000+ people. Can't be right. 
        self.cum_sum_work_pop = np.cumsum(self.ordered_work_pop[1:])  # remove the first work which are the people with no workplace
        self.work_population = self.cum_sum_work_pop[-1]
        self.nb_workplaces = len(self.cum_sum_work_pop)
        self.largest_workplaces_vaccinated = self.ordered_work_ids[1:self.nb_top_workplaces_vaccinated]

        print("******* ENTER rank_workplaces  *********")
        print("work_pop: ", self.ordered_work_pop[0:10])
        print("cum_sum, top 10: ", self.cum_sum_work_pop[0:10])
        print("rank_workplaces: self.nb_workplaces= ", self.nb_workplaces)
        print("* total work population: ", self.work_population)
        print("... nb_top_workplaces_vaccinated: ", self.nb_top_workplaces_vaccinated)  # should be integer
        print("* total work population to vaccinate: ", self.cum_sum_work_pop[self.nb_top_workplaces_vaccinated])
        print("* work_id[0]: ", self.ordered_work_ids[0])
        print("******* EXIT rank_workplaces *********")

    #--------------------------------------------
    def setupMaskReduction(self, avg, cv):
        """
        :param avg: Float
        Average reduction in mask efficiency. A mask efficiency reduction of zero leaves the default edge weight unchanged
        Efficiencies are in [0,1]
        :param cv: Float
        Coefficient of variation = std / avg
        Mask efficiencies (weight = (1-mask_reduction_A)*(1-mask_reduction_B) is a person-level quantity
        Every person carries (or not) a mask that retains its characteristics throughout the simulation
        """

        std = cv * avg
        reduction = np.random.normal(avg, std, self.population)
        print(self.population); quit()
        reduction[np.where(reduction < 0.)] = 0.
        reduction[np.where(reduction > 1.)] = 1.
        self.mask_reduction = reduction
        return reduction

    def setupSocialDistanceReduction(self, avg, cv):
        """
        :param avg: Average reduction in the efficacy of social distancing
        :param cv: Coefficient of variation = std / avg
        Social distance reduction reduction (weight = (1-mask_reduction) is a person-level quantity
        Every person social distances (or not) and retains this property across the simulation
        """

        std = cv * avg
        reduction = np.random.normal(avg, std, self.population)
        reduction[np.where(reduction < 0.)] = 0.
        reduction[np.where(reduction > 1.)] = 1.
        self.social_distancing_reduction = reduction
        return reduction
    #------------------
    # Called from the driver script
    def infectPopulace(self, perc):
        # vaccinate a fraction perc 
        """
        :param perc
        infect a fraction perc [0,1] of the population at random, all ages
        """

        infected_01 = bernoulli.rvs(perc, size=self.population)
        ## ERROR: list object self.populace has no keys()
        # how to get the keys of a list subject to conditionals. 
        self.initial_infected = np.where(infected_01 == 1)[0]
        #print("self.initial_infected= ", self.initial_infected); quit()

    #-------------------------------------------------
    # Why this more sophisticated version, with same signature as the two-line version above?
    def vaccinatePopulace(self, perc):
        # vaccinate a fraction perc 

        """
        :param perc
        Vaccinate a fraction perc [0,1] of the population at random, all ages
        """
        self.perc_populace_vaccinated = 0.0

        print("\n\n************ ENTER vaccinatePopulace ***************")

        self.initial_vaccinated = set()

        workplace_populace_vaccinated = []
        if self.nb_top_workplaces_vaccinated > 0:
            self.rank_workplaces()
            # Vaccinate the top n workplaces
            for i in range(1,self.nb_top_workplaces_vaccinated+1):
               people = self.workplaces[self.ordered_work_ids[i]]   # <<<<<<<
               workplace_populace_vaccinated.extend(people) # people is a list  ### MUST BE WRONG

        self.initial_vaccinated.update(workplace_populace_vaccinated)

        school_populace_vaccinated = []
        #if self.nb_top_schools_vaccinated > 0:
        if True:
            self.rank_schools()
            # Vaccinate the top n schools
            for i in range(1,self.nb_top_schools_vaccinated+1):
               people = self.schools[self.ordered_school_ids[i]]
               school_populace_vaccinated.extend(people)  # people is a list

        self.initial_vaccinated.update(school_populace_vaccinated)

        # if vaccinate the households of the workers in the largest workplaces
        # if vaccinate the households of the children in the largest schools

        if perc > 0.0001 and perc < 0.9999:
            self.perc_populace_vaccinated = perc
            vaccinated_01 = bernoulli.rvs(perc, size=self.population)
        elif perc > 0.99:
            self.perc_populace_vaccinated = 1.0
            vaccinated_01 = np.ones(self.population)
        elif perc < 0.01:
            self.perc_populace_vaccinated = 0.0
            vaccinated_01 = np.zeros(self.population)

        # self.populace has no keys. ???? BUG?
        #general_populace_vaccinated = np.asarray(list(self.populace.keys()))[vaccinated_01 == 1]
        general_populace_vaccinated = np.where(vaccinated_01 == 1)[0]
        print("general_populace_vaccinated= ", general_populace_vaccinated)
        print("self.initial_vaccinated= ", self.initial_vaccinated) # set
        #lll = list(general_populace_vaccinated[0])
        #print("general_populace_vaccinated= ", lll)
        self.initial_vaccinated.update(general_populace_vaccinated)

        self.workplace_population_vaccinated  = len(workplace_populace_vaccinated)
        self.school_population_vaccinated     = len(school_populace_vaccinated)
        self.initial_vaccinated_population    = len(self.initial_vaccinated)

        self.perc_workplace_vaccinated = self.nb_top_workplaces_vaccinated / self.nb_workplaces
        self.perc_school_vaccinated = self.nb_top_schools_vaccinated / self.nb_schools
        self.perc_people_vaccinated_in_workplaces = self.nb_top_schools_vaccinated / self.nb_schools

        print("after rank_workplaces, nb workplaces: ", self.nb_workplaces)
        print("* (cumsum) total work population to vaccinate: ", self.cum_sum_work_pop[self.nb_top_workplaces_vaccinated])
        print("vaccinatePopulace, nb people to vaccinate in the workplace: ", len(workplace_populace_vaccinated))

        print("* (cumsum) total school population to vaccinate: ", self.cum_sum_school_pop[self.nb_top_schools_vaccinated])
        # WRONG
        print("vaccinatePopulace, nb people to vaccinate in the schools: ", len(school_populace_vaccinated))

        print("*workplaces to vaccinate: ", self.nb_top_workplaces_vaccinated)
        print("*perc to vaccinate in workplaces: ", self.perc_people_vaccinated_in_workplaces)
        print("top of vaccinatePopulace: nb workplaces: ", self.nb_workplaces)

        print("Total initial population: ", self.population)
        #print("initial vaccinated: ", self.initial_vaccinated)

        if len(self.initial_vaccinated) != 0: 
            print("nb initial vaccinated: ", len(self.initial_vaccinated))
            print("fraction of general population vaccinated: ", len(self.initial_vaccinated) / self.population)

        print("nb schools vaccinated: ", self.nb_top_schools_vaccinated)

        print()

        # ERROR: nb of people in workplace to vaccinate MUST BE LESS than workplace population
        print("nb people in workplace to vaccinate: ", len(workplace_populace_vaccinated))
        print("fraction of workplace populace vaccinated: ", len(workplace_populace_vaccinated) / self.work_population)
        print("total workplace populace vaccinated: ", self.workplace_population_vaccinated)
        print("total workplace population: ", self.work_population)

        print()
  
        print("fraction of school populace vaccinated: ", len(school_populace_vaccinated) / self.school_population)
        print("total school population: ", self.school_population)
        print("total school populace vaccinated: ", self.school_population_vaccinated)
    
        print("self.nb_workplaces= ", self.nb_workplaces)
        print("self.nb_top_workplaces_vaccinated= ", self.nb_top_workplaces_vaccinated)

        print("Inside Vaccinate Populace")

        vacc = self.createVaccinationDict()
        for k,v in vacc.items():
            print("vacc[%s]: "%k, v)

        print("\n\n************ EXIT vaccinatePopulace ***************")
                    
    #----------------
    def printEnvironments(self):
        keys = list(self.environments.keys())
        envs = set()
        for k in keys:
            envs.add(self.environments[k].quality)


    #-------------------------------------------------
    def differentiateMasks(self, type_probs):
        """
        differentiateMasks picks a mask type, for each person in the population, as an int,
        and adds it to the populace graph

        :param type_probs: list
        the prob for each  mask type.
        should sum to one. Everybody should have a mask preference, regardless of whether they wear one

        :return:
        """
        #this picks a type of mask that each person uses, based on 'typeProbs'
        #test if it's a reasonable distribution

        assert int(round(sum(type_probs))) == 1, "invalid prob distribution, sum != 1"

        #pick masks randomly by probDistribution
        num_mask_types = len(type_probs)
        self.num_mask_types = num_mask_types
        #print("num mask types: ", num_mask_types); quit()  # 3

        for person in self.populace:
            #+1 because zero needs to represent unmasked
            person["mask_type"] = (np.random.choice(range(num_mask_types)))+1

    def __str__(self):
        """
        Describes the following qualities of the model:
        total population, total edges, total weight
        """
        str = "A PopulaceGraph Model: \n"
        #if isBuilt == False:

    #--------------------------
    def buildNetworks(self, netBuilder):
        """
        builds net for each environment, then,
        constructs a graph for the objects populace

        :param netBuilder: NetBuilder object
        defines how to choose edges and weights by each environment
        """
        #None is default so old scripts can still run. self not defined in signatur
        #for index in self.environments:
            #environment = self.environments[index]
            #environment.network(netBuilder)
        #self.isBuilt = True

        #None is default, so old scripts can still run. self not defined in signature
        for index in self.environments:
            environment = self.environments[index]
            environment.network(netBuilder)

        self.graph = nx.Graph()
        self.graph.add_nodes_from(list(range(len(self.populace))))
        print("len(populace): ", len(self.populace))
        print("self.population = ", self.population)
        #add the edges of each environment to a single networkx graph
        for environment in self.environments: self.graph.add_weighted_edges_from(self.environments[environment].edges, weight = "transmission_weight")
        self.isBuilt = True

    def reweight(self, netBuilder, new_prev_prevalences = None):
        """
        :param netBuilder: netBuilder object
        to calculate new weights
        :param new_prev_prevalences: dict
        to change the preventions used in each environment before reweight
        """

        #choose new preventions if requested
        for environment in self.environments:
            self.environments[environment].reweight(netBuilder, new_prev_prevalences)
    #merge environments, written for plotting and exploration
    def returnMergedEnvironments(self, env_indexes, partitioner = None):
        """
        :param env_indexes: environments to combine
        :param partitioner:  to partition the  new combo environment, optional
        :return:
        """

        if partitioner == None:
            partitioner = self.environments[env_indexes[0]]
        members = []
        for index in env_indexes:
            members.extend(self.environments[index].members)
        return StructuredEnvironment(None, members, 'multiEnvironment', self.populace, None, partitioner)

    def plotContactMatrix(self, p_env):
        '''
        This function plots the contact matrix for a partitioned environment
        :param p_env: must be a partitioned environment
        '''

        if p_env == None:
            p_env = self.returnMultiEnvironment()
        contact_matrix = self.returnContactMatrix(p_env)
        plt.imshow(contact_matrix)
        plt.title("Contact Matrix for members of {} # {}".format(p_env.quality, p_env.index))
        labels = p_env.partitioner.labels
        if labels == None:
            labels = ["{}-{}".format(5 * i, (5 * (i + 1))-1) for i in range(15)]
        axisticks= list(range(15))
        plt.xticks(axisticks, labels, rotation= 'vertical')
        plt.yticks(axisticks, labels)
        plt.xlabel('Age Group')
        plt.ylabel('Age Group')
        plt.show()

    #-------------------------------------------------------------------
    #def SIRperBracket(self, tlast): 
    def SIRperBracket(self, age_statuses): 
        # Collect the S,I,R at the last time: tlast
        # print("tlast.R= ", list(tlast.keys())); 
        # Replace 'S', 'I', 'R' by [0,1,2]

        #print("len(age_statuses)= ", len(age_statuses))
        age_groups = self.pops_by_category["age_groups"]
        count = 0
        for b,n in age_groups.items():
            count += len(n)
        #print("total number of nodes in all age brackets: ", count)

        brackets = {}
        count = 0
        for bracket, nodes in age_groups.items():
            #print("GE: bracket= ", bracket)
            # nodes in given age bracket
            b = brackets[bracket] = []
            #print("age bracket: ", bracket)
            #print("age bracket nodes= ", nodes)
            for n in nodes:
                #print("GE: n= ", n)
                try:
                    b.append(age_statuses[n])  # S,I,R
                except:
                    print("except, key: n= ", n)  # I SHOULD NOT END UP HERE
                    print("List of graph nodes with SIR statuses")
                    print("age_statuses.keys: ", list(age_statuses.keys()))
                    quit()

            count += len(nodes)
            #print("count= ", count, ",  bracket= ", bracket)
            
        ages_d = {}
        for bracket in age_groups.keys():
            blist = brackets[bracket]
            ages_d[bracket] = {'S':0, 'I':0, 'R':0}  # nb S, I, R
            for s in blist:
                ages_d[bracket][s] += 1
        #print("inside SIRperBracket")
        #print("  keys(ages_d): ", list(ages_d.keys()))
        return ages_d

    #-------------------------------------------------------------------
    def simulate(self, gamma, tau, simAlg=EoN.fast_SIR, title=None, full_data=True, preventions=None, global_dict={}):

        # Gordon Change
        #print("graph nodes: ", self.graph.number_of_nodes())
        #print("graph edges: ", self.graph.number_of_edges())
        assert self.graph.number_of_nodes() > 0, "nb graph nodes should be positive"
        assert self.graph.number_of_edges() > 0, "nb graph edges should be positive"

        self.global_dict = global_dict

        #simResult = simAlg(self.graph, tau, gamma, rho = 0.001, transmission_weight='transmission_weight', return_full_data=full_data)
        print("initial_infected: ", self.initial_infected)
        print("initial_recovered: ", self.initial_recovered)
        simResult = simAlg(self.graph, tau, gamma, initial_recovereds=self.initial_recovered, initial_infecteds=self.initial_infected, transmission_weight='transmission_weight', return_full_data=full_data)
        #self.sims.append([simResult, title, [gamma, tau], preventions])

        """
        graph = nx.Graph()
        #add the edges of each environment to a single networkx graph
        print("total nb environments: ", len(self.environments)); 
        for environment in self.environments: 
            graph.add_weighted_edges_from(self.environments[environment].edges, weight = "transmission_weight")
        print("Before simlation: Graph: nb nodes: ", graph.number_of_nodes())

        simResult = simAlg(graph, tau, gamma, rho = 0.001, transmission_weight='transmission_weight',return_full_data=full_data)
        #self.sims.append([simResult, title, [gamma, tau], preventions])
        """

        start2 = time.time()
        sr = simResult
        statuses = {}
        last_time = simResult.t()[-1]
        print("last_time= ", last_time) # 132 190.8

        for tix in range(0, int(last_time)+2, 2):
            # statuses[tix]: for each node of the graph, S,I,R status
            statuses[tix] = sr.get_statuses(time=tix)
        key0 = list(statuses.keys())[0]
        # statuses[graph node] = Dictionary: node# => 'S', 'I', or 'R'}

        self.record.print("handle simulation output: {} seconds".format(time.time() - start2))

        # Next: calculate final S,I,R for the different age groups. 

        start3 = time.time()

        ages_d = {}

        for k,v in statuses.items():
            ages_d[k] = self.SIRperBracket(v)

            """
            for bracket in ages_d[k].keys():  # bracket is either integer or string. How to change? 
                #print("bracket: ", bracket)
                #print("   keys: ", list(ages_d[k].keys()))
                counts = ages_d[k][bracket]
                print("bracket: ", bracket, ",  counts[S,I,R]: ", bracket, counts['S'], counts['I'], counts['R'])
            """
            
        self.record.print("time to change 'S','I','R' to 0,1,2 for faster processing: %f sec" % (time.time()-start3))
        #-----------
        # Create a dictionary to store all the data and save it to a file 
        data = {}
        u = Utils()
        SIR_results = {'S':sr.S(), 'I':sr.I(), 'R':sr.R(), 't':sr.t()}
        SIR_results = u.interpolate_SIR(SIR_results)
        data['SIR'] = SIR_results
        data['title'] = title
        data['params'] = {'gamma':gamma, 'tau':tau}
        data['preventions'] = self.preventions
        data['prevention_reductions'] = self.prevention_reductions  # no longer needed
        data['prevention_prevalences'] = self.prevention_prevalences
        data['ages_SIR'] = ages_d # ages_d[time][k] ==> S,I,R counts for age bracket k
        data['vacc_dict'] = self.createVaccinationDict()
        data['global_dict'] = self.global_dict

        x = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "%s, gamma=%s, tau=%s, %s" % (title, gamma, tau, x)
        print("saveResults: filename: ", filename)
        self.saveResults(filename, data)

    #-------------------------------------------
    def saveResults(self, filename, data_dict):
        """
        :param filename: string
        File to save results to
        :param data_dict: dictionary
        Save SIR traces, title, [gamma, tau], preventions
        # save simulation results and metadata to filename
        """

        full_path = "/".join([self.basedir, filename])

        with open(full_path, "wb") as pickle_file:
            pickle.dump(data_dict, pickle_file)

        """
        # reload pickle data
        fd = open(filename, "rb")
        d = pickle.load(fd)
        SIR = d['sim_results']
        print("SIR['t']= ", SIR['t'])
        quit()
        """


    #---------------------------------------------------------------------------
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
            plt.title("Degree plot for members of {} # {}".format(environment.quality, environment.index))
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
        plt.savefig(self.basedir+"/plotNodeDegreeHistogram.pdf")


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

    def getPeakPrevalences(self):
        return [max(sim[0].I()) for sim in self.sims]

    #If a partitionedEnvironment is specified, the partition of the environment is applied, otherwise, a partition must be passed
    def plotBars(self, environment = None, SIRstatus = 'R', normalized = False):
        """
        Will show a bar chart that details the final status of each partition set in the environment, at the end of the simulation
        :param environment: must be a partitioned environment
        :param SIRstatus: should be 'S', 'I', or 'R'; is the status bars will represent
        :param normalized: whether to plot each bar as a fraction or the number of people with the given status
        #TODO finish implementing None environment as entire graph
        """
        partition = environment.partitioner
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
        plt.savefig(self.basedir+"/evasionChart.pdf")

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
    def __init__(self, basedir):
        self.log = ""
        self.comments = ""
        self.graph_stats = {}
        self.last_runs_percent_uninfected = 1
        self.basedir = basedir

        try:
            mkdir(self.basedir)
        except:
            pass

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
        #log_txt = open("./ge_simResults/{}/log.txt".format(self.timestamp), "w+")
        log_txt = open(basedir+"/log.txt", "w+")
        log_txt.write(self.log)
        if self.comments != "":
            #comment_txt = open("./ge_simResults/{}/comments.txt".format(self.timestamp),"w+")
            comment_txt = open(basedir+"/comments.txt", "w+")
            comment_txt.write(self.comments)

#written by Gordon
class Utils:
    def interpolate_SIR(self, SIR):
        S = SIR['S']
        I = SIR['I']
        R = SIR['R']
        t = SIR['t']
        print("len(t)= ", len(t))
        # interpolate on daily intervals.
        new_t = np.linspace(0., int(t[-1]), int(t[-1])+1)
        func = interp1d(t, S)
        Snew = func(new_t)
        func = interp1d(t, I)
        Inew = func(new_t)
        func = interp1d(t, R)
        Rnew = func(new_t)
        #print("t= ", new_t)
        #print("S= ", Snew)
        #print("I= ", Inew)
        #print("R= ", Rnew)
        SIR['t'] = new_t
        SIR['S'] = Snew
        SIR['I'] = Inew
        SIR['R'] = Rnew
        return SIR
