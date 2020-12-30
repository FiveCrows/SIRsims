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
from  synthModule import *
from scipy.interpolate import interp1d
from scipy.stats import bernoulli
from collections import defaultdict


class NetBuilder:
    """
    This class is written to hold network building algorithms
    The Netbuilder can be called to create a densenets for Environment
    and Random nets for partitioned Environments.
    """

    def __init__(self,  avg_contacts = None):
        """
        :param env_type_scalars: dict
        each environment type must map to a float. is for scaling weights

        :param avg_contacts:
        if specified, the number of edges picked for an environment will be chosen to meet avg_contacts
        """
        self.global_weight = 1

    def addEdge(self, nodeA, nodeB, environment):
        """
        makes sure the edge is added in the proper direction
        :return:
        """

        # the conditional ensures that (nodeA,nodeB) and (nodeB,nodeA) are
        # not separate edges in the environment edge list. This avoid errors when
        # maintaining lists constructed outside the Networkx library.

        if nodeA < nodeB:
            environment.addEdge(nodeA, nodeB)
        else:
            environment.addEdge(nodeB, nodeA)
            pass

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
                environment.addEdge(nodeA, nodeB)


    def genRandEdgeList(self, setA, setB, n_edges):
        if n_edges == 0:
            return []
        n_edges = int(n_edges)
        n_A = len(setA)
        n_B = len(setB)
        if setA == setB:
            pos_edges = n_A * (n_A - 1) / 2
        else:
            pos_edges = n_A * n_B

        #p_duplicate = n_edges/pos_edges
        #if p_duplicate> 0.5:
        #pass
            #list = random.shuffle(enumerate())
        edge_dict = {}
        while len(edge_dict) < n_edges:
            A, B = random.choice(setA), random.choice(setB)
            if A > B:
                edge_dict[A, B] = 1
            elif B > A:
                edge_dict[B, A] = 1
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
                    nodeA, nodeB = A[i], B[(begin_B_edges +j)%size_B]
                    #weight = self.getWeight(nodeA, nodeB, environment)
                    weight = 1.0 # GE
                    environment.addEdge(nodeA,nodeB)
                else:
                    remainder +=1


        eList = self.genRandEdgeList(members_A, members_B, remainder)
        for edge in eList:
            #weight = self.getWeight(edge[0], edge[1], environment)
            weight = 1.0 # GE
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
         if avg_
        degree is not None, then the contact matrix should scaled such that the average degree

        :param topology: string
        can be either 'random', or 'strogatz'

        :return:
        """

        #make sure there's at least 1 edge to add
        if environment.population <= 1:
            return

        #to clean up code just a little
        environment.edges = []
        p_sets = environment.partition
        CM = environment.returnReciprocatedCM()

        """
        #add gaussian noise to contact matrix values
        if "contact" in self.cv_dict: 
            CM = CM*np.random.normal(1, self.cv_dict["contact"], CM.shape)
            CM[np.where(CM < 0.)] = 0.
        """

        #assert isinstance(environment, StructuredEnvironment), "must be a structured environment"
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
        
        if avg_degree == 0: return

        #print('by the sum of the CM, avg_degree should be : {}'.format(avg_degree ))
        #determine total edges needed for entire network. There are two connections per edge)
        total_edges = math.floor(avg_degree * environment.population/2)
        #print("avg_degree= ", avg_degree, " env pop: ", environment.population)

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
                    # GE: should divide by 2. This is undirected graph
                    max_edges = p_n[i] * (p_n[i]-1)/2
                    if max_edges <= num_edges:
                        self.buildDenseNet(environment)
                        continue
                else:
                    num_edges = int(total_edges*contactFraction*2)
                    max_edges = p_n[i] * p_n[j]
                    if num_edges > max_edges:
                        num_edges = max_edges
                if num_edges == 0:
                    continue
                expected_degree = num_edges/p_n[i]
                if expected_degree>10:
                    print(expected_degree)
                
                #to  compensate by scaling the weights up a bit
                #residual_scalar = total_edges * contactFraction / num_edges
                #if residual_scalar>2 and sizeA>3:
                    #print("error in environment # {}, it's contacts count for i,j = {} is {}but there are only {} people in that set".format(environment.index, index_i, CM[index_i,index_j], len(environment.partitioned_members[index_i])))
                edgeList = self.genRandEdgeList(p_sets[i], p_sets[j], num_edges)
                for edge in edgeList:
                    self.addEdge(edge[0], edge[1], environment)

    def setEnvScalars(self, env_scalars):
        self.env_scalars = env_scalars

    #def setPreventions(self, preventions):
        #self.preventions = preventions


    #def weightEnvs(self, envs, ):


class StrogatzNetBuilder(NetBuilder):
    # NOT USED

    def netStrogatz(self, environment,  num_edges, weight_scalar = 1, subgroup = None, rewire_p = 0.1):
        """
         netStrogatz creates a strogatz net in the given environment

         :param environment: Environment object,
        where to add edges
         :param num_edges: int
         the number of edges to add to the environment
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
            self.buildDenseNet(environment)
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

class prefAttachmentNetBuilder(NetBuilder):
    pass

class PopulaceGraph:
    """
    A list of people, environments, and functions, for tracking a weighted graph to represent contacts between members of the populace
    """

    def __init__(self, slim = False, timestamp=None):
        """        
        :param partition: Partitioner
        needed to build schools and workplaces into structured environments
        :param attributes:
        names for the characteristics to load for each person
        :param slim: bool
        if set to True, will filter 90% of people. Mainly for debugging
        :param prevention_adoptions: dict
        keys should be 'household', 'school', or 'workplace'. Each should map to another dict,
        with keys for 'masking', and 'distancing', which should map to an int in range[0:1] that represents
        the prevelance of people practicing the prevention strategy in the environment
        """

        self.isBuilt = False
        self.sims = []
        self.contactMatrix = None
        self.total_weight = 0
        self.total_edges = 0
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

        self.basedir = "./results/{}".format(self.timestamp)
        print("basedir= ", self.basedir)
        
        self.preventions = None

        if slim == True:
            file = open("./LeonCountyData/slimmedLeon.pkl", 'rb')            
        else:            
            file = open("./LeonCountyData/leon.pkl", 'rb')

        self.loadPopulace(file)

        # pick who masks and distances, in each environment
        # One has a list of people in each environment
        self.resetVaccinatedInfected()

        # must call once in constructor
    def loadPopulace(self, file):
        '''
        this function takes a pickled dict and loads all key:pair as object variables
        :param file: a file object to reference the pickle         
        '''
        #load synthetic data from file
        pickleDict = (pickle.load(file))
        self.partitioner = pickleDict.pop('partitioner')
        self.populace = pickleDict.pop('populace')
        self.pops_by_category = pickleDict.pop('pops_by_category')
        self.environments = pickleDict.pop('environments')
        self.__dict__.update(pickleDict)
        
        #make a default dict, to handle none case
        self.population = len(self.populace)
        self.schools = self.pops_by_category["school_id"]
        self.workplaces = self.pops_by_category["work_id"]
        #add direct reference to env objects in populace
        #None is temporarily added for cases where there is no school or workplace id
        #self.environments[None] = None
#outdated code, may be replaced later but not likely        
#        names = zip(['sp_hh_id', 'work_id', 'school_id'], ['household', 'workplace', 'school'])
#        for namePair in names:            
#            list(map(lambda x: x.update({namePair[1]: self.environments[x[namePair[0]]]}), self.populace))
        #None must be removed because None can't be networked later
#        self.environments.pop(None)

    #-------------------------------------------------
    def printEnvironments(self):
        keys = list(self.environments.keys())
        envs = set()
        for k in keys:
            envs.add(self.environments[k].env_type)

        # Environment type can be "household", "workplace", "school"
        #print("envs= ", envs)
        # attributers of environments[keys[1]]: "index', 'members', 'population', 'preventions', 'type'"
        #print(dir(self.environments[keys[1]]))
        for k in range(25000,25200):
            print("-----------------")
            print(self.environments[keys[k]].members)  # list of one element [12]. Meaning?
            print(self.environments[keys[k]].population)  # 1  (nb of members)
            print(self.environments[keys[k]].preventions)  # None (or a list?
            print(self.environments[keys[k]].env_type)  # list of one element [12]. Meaning?

    #-------------------------------------------------
    def resetVaccinatedInfected(self):
        # By default nobody in the population is recovered. 
        # Vaccination is modeled by setting a person's status to recovered

        # Reset to not vaccinate anybody anywhere 
        self.setNbTopWorkplacesToVaccinate(0, 0.)
        self.setNbTopSchoolsToVaccinate(0, 0.)

        # Rank locations. Must be called after setting nb places to vaccinate
        #print("** resetVaccinatedInfected")
        self.rankWorkplaces()
        self.rankSchools()

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

    def zeroWeights(self, env):
        # GE: How to turn off a school without reconstructing the school graph? 
        # NOT USED
        # Run through the edges and set the weight to zero
        start_time = time.time()
        G = self.graph
        for e in G.edges():
            environment = self.environments[self.graph.adj[e[0]][e[1]]['environment']]
            print("env, envi= ", env, environment.env_type)
            if env == environment.env_type:
                self.graph.adj[e[0]][e[1]]['transmission_weight'] = 0
        print("zeroWeights[%s]: %f sec" % (env, time.time() - start_time))

    def printWeights(self):
        G = self.graph
        for e in G.edges():
            w = G[e[0]][e[1]]['transmission_weight']
            env = self.environments[G[e[0]][e[1]]['environment']]
            print("weight(e): ", e, " => ", w, env.env_type)

    def setNbTopWorkplacesToVaccinate(self, nb, perc):
        self.nb_top_workplaces_vaccinated = nb
        self.perc_people_vaccinated_in_workplaces = perc
        print("*workplaces to vaccinate: ", self.nb_top_workplaces_vaccinated)
        print("*perc to vaccinate in workplaces: ", self.perc_people_vaccinated_in_workplaces)

    #---------------------
    def setNbTopSchoolsToVaccinate(self, nb, perc):
        self.nb_top_schools_vaccinated = nb
        self.perc_people_vaccinated_in_schools = perc
        print("self.nb_top_schools_vaccinated: ", nb)

    #------------------
    def rankSchools(self):
        # produces list of pairs (school_id, list of people ids: 0 through max_peoople )
        # replace the list of people ids by its length
        ordered_schools = sorted(self.schools.items(), key=lambda x: len(x[1]), reverse=True)

        # list of (school_id, school_structure), removing the 1st element, which is not a school. 
        ordered_schools = list(map(lambda x: [x[0], len(x[1])], ordered_schools))[1:]

        self.ordered_school_ids = [o[0] for o in ordered_schools[:]]
        self.ordered_school_pop = [o[1] for o in ordered_schools[:]]

        # Cumulative sum of school lengths
        # Sum from 1 since 0th index is a school with 200,000 students. Can't be right. 
        self.cum_sum_school_pop = np.cumsum(self.ordered_school_pop)
        self.school_population = self.cum_sum_school_pop[-1]
        self.nb_schools = len(self.cum_sum_school_pop)  
        self.largest_schools_vaccinated = self.ordered_school_ids[0:self.nb_top_schools_vaccinated]
        
        print("******* ENTER rankSchools *********")
        print("school_pop: ", self.ordered_school_pop[0:10]) #10  largest to smallest
        print("school_ids: ", self.ordered_school_ids[0:10])
        print("cum_sum, top 10: ", self.cum_sum_school_pop[0:10])
        print("rankSchools: self.nb_schools= ", self.nb_schools)
        print("* total school population: ", self.school_population)
        print("self.nb_top_schools_vaccinated: ", self.nb_top_schools_vaccinated)
        if self.nb_top_schools_vaccinated == 0:
            print("* total school population to vaccinate: 0")
        else:
            print("* total school population to vaccinate: ", self.cum_sum_school_pop[self.nb_top_schools_vaccinated-1])

    #----------------------------------
    def rankWorkplaces(self):

        # produces list of pairs (workplace is, list of people is)
        # replace the list of people ids by its length

        ordered_workplaces = sorted(self.workplaces.items(), key=lambda x: len(x[1]), reverse=True)
        ordered_workplaces = list(map(lambda x: [x[0], len(x[1])], ordered_workplaces))[1:]

        self.ordered_work_ids = [o[0] for o in ordered_workplaces[:]]
        self.ordered_work_pop = [o[1] for o in ordered_workplaces[:]]

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
        print("* work_id[0]: ", self.ordered_work_ids[0])

        if self.nb_top_workplaces_vaccinated == 0:
            print("* total workplace population to vaccinate: 0")
        else:
            print("* total workplace population to vaccinate: ", self.cum_sum_work_pop[self.nb_top_workplaces_vaccinated-1])

    #--------------------------------------------


    #--------------------------------------------
    def setupMaskingReductions(self, avg_std):
        """
        :param avg: list
        the average and the std in a list
        """
        avg, std = avg_std
        reduction = np.random.normal(avg, std, self.population)
        reduction[np.where(reduction < 0.)] = 0.
        reduction[np.where(reduction > 1.)] = 1.
        self.mask_reductions = reduction



    #-----------------------------------------
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

        #print("\n\n************ ENTER vaccinatePopulace ***************")

        self.initial_vaccinated = set()

        workplace_populace_vaccinated = []
        if self.nb_top_workplaces_vaccinated > 0:
            self.rankWorkplaces()
            # Vaccinate the top n workplaces
            for i in range(1,self.nb_top_workplaces_vaccinated+1):
               people = self.workplaces[self.ordered_work_ids[i]]   # <<<<<<<
               # Vaccinate the people in the workplace
               perc_work = self.perc_people_vaccinated_in_workplaces
               vaccinated_01 = bernoulli.rvs(perc_work, size=len(people))
               workplace_vaccinated = np.where(vaccinated_01 == 1)[0]
               #print("len(people): ", len(people))
               #print("len(vaccinated_01): ", len(vaccinated_01))
               peep = np.asarray(people)
               try:
                   workplace_populace_vaccinated.extend(peep[vaccinated_01 == 1]) 
               except:
                   print("Except:")
                   print("perc_work= ", perc_work)
                   print("vaccinated_01= ", vaccinated_01)
                   print("np.where(vaccinated_01 == 1):", np.where(vaccinated_01 == 1))
                   print("workplace_vaccinated: ", workplace_vaccinated)
                   print("people= ", people)
                   print("peep=np.asarray(people)= ", np.asarray(people))
                   print("peep= ", peep)
                   print("len(peep)= ", len(peep))
                   ppp = peep[vaccinated_01 == 1]
                   print("ppp= ", ppp)
                   quit()

        print("bef self.initial_vaccinated: ", len(self.initial_vaccinated))
        print("workplace_populace_vaccinated: ", len(workplace_populace_vaccinated))

        self.initial_vaccinated.update(workplace_populace_vaccinated)
        print("aft self.initial_vaccinated: ", len(self.initial_vaccinated))

        school_populace_vaccinated = []
        #if self.nb_top_schools_vaccinated > 0:
        if True:
            self.rankSchools()
            # Vaccinate the top n schools
            for i in range(1,self.nb_top_schools_vaccinated+1):
               people = self.schools[self.ordered_school_ids[i]]
               perc_school = self.perc_people_vaccinated_in_school
               vaccinated_01 = bernoulli.rvs(perc_school, size=len(people))
               school_vaccinated = np.where(vaccinated_01 == 1)[0]
               school_populace_vaccinated.extend(np.asarray(people)[vaccinated_01 == 1]) # people is a list  ### MUST BE WRONG

        self.initial_vaccinated.update(school_populace_vaccinated)

        # if vaccinate the households of the workers in the largest workplaces
        # if vaccinate the households of the children in the largest schools

        if perc > 0.0001 and perc < 0.9999:
            self.perc_populace_vaccinated = perc
            vaccinated_01 = bernoulli.rvs(perc, size=self.population)
        elif perc >= 0.99:
            self.perc_populace_vaccinated = 1.0
            vaccinated_01 = np.ones(self.population)
        elif perc < 0.01:
            self.perc_populace_vaccinated = 0.0
            vaccinated_01 = np.zeros(self.population)

        print("bef self.initial_vaccinated: ", len(self.initial_vaccinated))
        general_populace_vaccinated = np.where(vaccinated_01 == 1)[0]
        self.initial_vaccinated.update(general_populace_vaccinated)
        print("general_populace_vaccinated: ", len(general_populace_vaccinated))
        print("aft self.initial_vaccinated: ", len(self.initial_vaccinated))

        self.workplace_population_vaccinated  = len(workplace_populace_vaccinated)
        self.school_population_vaccinated     = len(school_populace_vaccinated)
        self.initial_vaccinated_population    = len(self.initial_vaccinated)

        print("nb vaccinated in pop, sch, wrk: %d, %d, %d\n" % (self.initial_vaccinated_population, self.school_population_vaccinated, self.workplace_population_vaccinated))

        self.perc_workplace_vaccinated = self.nb_top_workplaces_vaccinated / self.nb_workplaces
        self.perc_school_vaccinated = self.nb_top_schools_vaccinated / self.nb_schools
        self.perc_people_vaccinated_in_workplaces = self.nb_top_schools_vaccinated / self.nb_schools

        """ Put in a function?
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
        """ 
        vacc = self.createVaccinationDict()
                    
    #----------------
    def printEnvironments(self):
        keys = list(self.environments.keys())
        envs = set()
        for k in keys:
            envs.add(self.environments[k].env_type)


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
    def networkEnvs(self, netBuilder):
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
        self.isNetworked = True

        print("End of networkEnvs:")
        print("Graph parameters:")
        print("  - Node count: ", self.graph.number_of_nodes())
        print("  - Edge count: ", self.graph.number_of_edges())


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

    #----------------------------------------
    
    #------------------------------
    def simulate(self, gamma, tau, simAlg=EoN.fast_SIR, title=None, full_data=True, global_dict={}):

        self.global_dict = global_dict
        simResult = simAlg(self.graph, tau, gamma, initial_recovereds=self.initial_vaccinated, initial_infecteds=self.initial_infected, transmission_weight='transmission_weight', return_full_data=full_data)
        self.sims.append([title,simResult])

        start2 = time.time()
        sr = simResult
        statuses = {}
        last_time = simResult.t()[-1]

        for tix in range(0, int(last_time)+2, 2):
            # statuses[tix]: for each node of the graph, S,I,R status
            #print("sr= ", sr)
            #print("tix= ", tix)
            statuses[tix] = sr.get_statuses(time=tix)
        key0 = list(statuses.keys())[0]
        # statuses[graph node] = Dictionary: node# => 'S', 'I', or 'R'}

        

        # Next: calculate final S,I,R for the different age groups. 

        start3 = time.time()

        ages_d = {}

        for k,v in statuses.items():
            ages_d[k] = self.SIRperBracket(v)

        
        #-----------
        # Create a dictionary to store all the data and save it to a file 
        data = {}
        u = Utils()
        SIR_results    = {'S':sr.S(), 'I':sr.I(), 'R':sr.R(), 't':sr.t()}
        SIR_results    = u.interpolateSIR(SIR_results)
        data['SIR']    = SIR_results
        data['title']  = title
        data['params'] = {'gamma':gamma, 'tau':tau}
        data['preventions'] = self.preventions
        data['prevention_adoptions']  = self.prevention_adoptions
        data['ages_SIR']    = ages_d # ages_d[time][k] ==> S,I,R counts for age bracket k
        data['vacc_dict']   = self.createVaccinationDict()
        data['global_dict'] = self.global_dict
        data['initial_nb_recovered'] = len(self.initial_vaccinated)
        data['initial_nb_infected']  = len(self.initial_infected)

        x = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "%s, gamma=%s, tau=%s, %s" % (title, gamma, tau, x)
        print("saveResults: filename: ", filename)
        self.saveResults(filename, data)

    #------------------------------
    def weightNetwork(self, env_type_scalars, prevention_adoptions, prevention_efficacies):
        '''
        :param env_type_scalars: dict
        to associate each env type with a parameter for scaling its use
        :param prevention_adoptations: dict
        to associate a rate for mask and distancing practices to each environment type
        :param prevention_efficacies: list
        efficetiveness from 0 to 1 100% effective of the masking and distancing, and the std for these per unit
        :return:
        '''
        self.prevention_adoptions = prevention_adoptions
        assert self.isNetworked, "env must be networked before graph can be built"

        #draw an effictiveness of each persons practice of each prevention, then, for each environment, 
        #draw who is actually practices each prevention. This way, while people may practice a prevention 
        #within some environments, but not others, their effictiveness remains a constant across each env
        
        self.setupMaskingReductions(prevention_efficacies["masking"])
        
        #dispersal ={np.random.gamma(1,0)}#is a stub, will be replaced with better function later
        envs = self.environments
        for env in envs.values():
            edges = env.edges
            # draw masks and distancers
            env.drawPreventions(prevention_adoptions, self.populace)
            #distribute the reduction factor for distancing on each edge
            avg,std = prevention_efficacies["distancing"]
            dist_reduction = np.random.normal(avg, std, len(env.edges))            
            dist_reduction[np.where(dist_reduction < 0.)] = 0.
            dist_reduction[np.where(dist_reduction > 1.)] = 1.

            for ix, edge in enumerate(edges):
                #get distancing factor
                dist_wf = 1- env.distancing_adoption[edge] * dist_reduction[ix]                
                #get mask factor
                pa, pb = edge
                mask_wf = ((1 - env.mask_adoption[pa] * self.mask_reductions[pa]) 
                         * (1 - env.mask_adoption[pb] * self.mask_reductions[pb]))                
                #add weighted edge                
                weight = mask_wf * dist_wf * env_type_scalars[env.env_type] #* dispersal #* dist_wf
                self.graph.add_edge(pa, pb, transmission_weight = weight)

            #print("mask_weight_factor= ", mask_weight_factor)

    # self.mask_reductions: defined for every person of the graph
        # Whether a mask is worn or not are percentages set for each environment type

        print("Finished with weightNetwork.")
        print("Graph parameters:")
        print("  - Node count: ", self.graph.number_of_nodes())
        print("  - Edge count: ", self.graph.number_of_edges())

    #-------------------------------------------


    #---------------------------------------------------------------------------

    def reset(self):
        self.sims = []
        self.graph = nx.Graph()
        self.total_weight = 0
        self.total_edges = 0


