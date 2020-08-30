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



class Environment:
    def __init__(self, members, type, masking):
        self.members = members
        self.type = type
        self.masking = masking

       # self.distancing = distancing

class Partition:
    def __init__(self, enumerator, attribute, names = None):
        self.enumerator = enumerator
        self.attribute = attribute
        self.names = names
        self.attribute_values = dict.fromkeys(set(enumerator.values()))
        self.num_sets = len(enumerator)

class PartitionedEnvironment(Environment):
    def __init__(self, members, type, masking, populace, contact_matrix, partition):
        super().__init__(members, type, masking)
        self.partition = partition
        self.contact_matrix = contact_matrix
        self.id_to_partition = dict.fromkeys(members)
        self.partitioned_members = {i:[] for i in range(partition.num_sets)}
        for person in members:
            #determine the number for which group the person belongs in, depending on their attribute
            group = partition.enumerator[populace[person][partition.attribute]]
            #add person to  to dict in group
            self.partitioned_members[group].append(person)

        for set in self.partitioned_members:
            for person in self.partitioned_members[set]:
                self.id_to_partition[person] = (set)


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


class PopulaceGraph:
    def __init__(self, weighter, environment_degrees, environment_masking =  {'work': 0, 'school':0}, partition = None, graph = None, populace = None, pops_by_category = None, categories = ['sp_hh_id', 'work_id', 'school_id', 'race', 'age'], slim = False):
        self.trans_weighter = weighter
        self.isBuilt = False
        #self.record = Record()
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
        for household in households:
            houseObject = Environment(households[household], "household", 0)
            self.environments[household] = (houseObject)

        #adding workplaces to environment list
        workplaces = self.pops_by_category["work_id"]
        with open("../ContactMatrices/Leon/ContactMatrixWorkplaces.pkl", 'rb') as file:
            work_matrices = pickle.load(file)
        for place in workplaces:
            if place != None:
                workplace = PartitionedEnvironment(workplaces[place], "workplace", environment_masking['work'], self.populace, work_matrices[place], partition )
                self.environments[place] = (workplace)


        schools = self.pops_by_category["school_id"]
        with open("../ContactMatrices/Leon/ContactMatrixSchools.pkl", 'rb') as file:
            school_matrices = pickle.load(file)
        for place in schools:
            if place != None:
                school = PartitionedEnvironment(schools[place], "school", environment_masking['school'], self.populace, work_matrices[place], partition )
                self.environments[place] = (school)

        print("stop")


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



