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
    def __init__(self, enumerator, attribute, names = None):
        self.enumerator = enumerator
        self.attribute = attribute
        self.names = names
        self.attribute_values = dict.fromkeys(set(enumerator.values()))
        self.num_sets = (len(np.unique(list(enumerator.values()))))



    def partitionGroup(self, members, populace):
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
    def __init__(self, index, members, type, preventions = None):
        self.index = index
        self.members = members
        self.type = type
        self.preventions = preventions
        self.population = len(members)
       # self.distancing = distancing


class PartitionedEnvironment(Environment):
    def __init__(self, index, members, type, populace, contact_matrix, partition, preventions = None):
        super().__init__(index,members, type, preventions)
        self.partition = partition
        self.contact_matrix = contact_matrix
        self.id_to_partition = dict.fromkeys(members)

        #self.total_matrix_contact = contact_matrix.sum()
        self.partitioned_members = partition.partitionGroup(members, populace)
        for set in self.partitioned_members:
            for person in self.partitioned_members[set]:
                self.id_to_partition[person] = (set)


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
        if random.random() < environment.preventions["masking"]:
            weight = weight * self.prevention_reductions["masking"]
            if random.random() < environment.preventions["masking"]**2:
                weight = weight * self.prevention_reductions["masking"]
        #distancing weight reduction
        weight = weight*(1-(1-self.prevention_reductions["distancing"]) * environment.preventions["distancing"])
        return weight


class PopulaceGraph:
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

        #list households:

        #load contact_matrices and build environments
        with open("../ContactMatrices/Leon/ContactMatrixSchools.pkl", 'rb') as file:
            schoolCM = pickle.load(file)
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




    def build(self, weighter, preventions, env_degrees):
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
            self.addEnvironment(environment)
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
            members.extend ((self.environments[index].members))
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


    def addEnvironment(self, environment):
        if environment.type == 'household':
            self.clusterDense(environment)
        else:
            self.clusterPartitionedStrogatz(environment, self.environment_degrees[environment.type])


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

    #clusterBipartite is particularly written to be used in clusterStrogatzByContact
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
                    self.addEdge(A[i], B[B_side],environment, weight_scalar)
                else:
                    self.addEdge(random.choice(A), random.choice(B), environment, weight_scalar)

        for i in range(remainder):
            self.addEdge(random.choice(A), random.choice(B), environment, weight_scalar)

    def clusterPartitionedStrogatz(self, environment, avg_degree):
        assert isinstance(environment, PartitionedEnvironment), "must be a partitioned environment"
        totalEdges = math.floor(avg_degree * environment.population/2)
        matrix_sum = sum(sum(environment.contact_matrix))
        #default_weight = totalContact/totalEdges

        for groupA in environment.partitioned_members:
            sizeA = len(environment.partitioned_members[groupA])
            if sizeA == 0:
                continue
            for groupB in environment.partitioned_members:
                sizeB = len(environment.partitioned_members[groupB])
                if sizeB == 0:
                    continue
                weightFraction = environment.contact_matrix[groupA, groupB]/matrix_sum
                number_edges = int(totalEdges*weightFraction)
                if number_edges == 0:
                    continue

                if groupA == groupB:
                    max_edges = sizeA * (sizeA-1)/2
                    if max_edges < number_edges:
                        number_edges = max_edges
                    residual_scalar = totalEdges * weightFraction / number_edges
                    self.clusterStrogatz(environment, num_edges = number_edges, subgroup = environment.partitioned_members[groupA], weight_scalar = residual_scalar)

                else:
                    max_edges = sizeA*sizeB
                    if max_edges < number_edges:
                        number_edges = max_edges
                    residual_scalar = totalEdges * weightFraction / number_edges
                    self.clusterBipartite(environment, environment.partitioned_members[groupA],environment.partitioned_members[groupB], number_edges, weight_scalar = residual_scalar)

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
        partition = environment.partition
        contact_matrix = np.zeros([partition.num_sets, partition.num_sets])
        partition_sizes = np.zeros(partition.num_sets)
        for index in range(partition.num_sets):
            partition_sizes[index] = len(partition.enumerator)

        for i in environment.members:
            iPartition = environment.id_to_partition[i]
            try:
                iHomies = list(set(environment.members) & set(self.graph[i]))
            except:
                continue
            for j in iHomies:
                jPartition = environment.id_to_partition[j]
                contact_matrix[iPartition, jPartition] += self.graph[i][j]['transmission_weight'] / partition_sizes[iPartition]
        # plt.imshow(np.array([row / np.linalg.norm(row) for row in contact_matrix]))
        return contact_matrix


    def plotContactMatrix(self, environment):
        if environment == None:
            environment = self.returnMultiEnvironment()
        contact_matrix = self.returnContactMatrix(environment)
        plt.imshow(contact_matrix)
        plt.show()


    def plotNodeDegreeHistogram(self, environment = None):
        if environment != None:
            graph = self.graph.subgraph(environment.members)
            plt.title("Degree plot for members of {} # {}".format(environment.type, environment.index))
        plt.hist([degree[1] for degree in nx.degree(graph)])
        plt.ylabel("total people")
        plt.xlabel("degree")
        plt.show()
        plt.savefig("./simResults/{}/".format(self.record.stamp))


    def plotSIR(self, memberSelection = None):
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
    def plotBars(self, environment = None, SIRstatus = 'R'):
        partition = environment.partition
        if isinstance(environment, PartitionedEnvironment):
            partitioned_people = environment.partitioned_members
            partition = environment.partition
        else:
            assert partition != None
            partitioned_people = partition.partitionGroup(self.populace.keys(), self.populace)
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
                element = partitioned_people[index]
                if len(element) == 0:
                    #no bar if no people
                    totals.append(0)
                    continue
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

    def getR0(self):
        sim = self.sims[-1]
        herd_immunity = list.index(max(sim.I))
        return(self.population/sim.S([herd_immunity]))

    def clearSims(self):
        self.sims = []

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



