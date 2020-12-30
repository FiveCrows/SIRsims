import random
import numpy as np 

class Person:
    def __init__(self,attrs: list):
        '''
            Initialize Person objects with attributes from array
        '''
        convertSex=lambda x: 1 if x=='F' else 0 #Sex: M=0, F=1
        convertCol=lambda y: None if y=="X" else y

        self.sp_id=int(attrs[0])
        self.sp_hh_id=int(attrs[1])
        self.age=int(attrs[2])
        self.sex=convertSex(attrs[3])
        self.race=int(attrs[4])
        self.relate=int(attrs[5])
        self.school_id=convertCol(attrs[6])
        self.work_id=convertCol(attrs[7])
        self.comorbidities={'Hypertension': False,'Obesity': False, 'Lung disease':False,'Diabetes': False,'Heart disease':False,
        'MaskUsage': False, 'Other': False}


class Environment:
    """
    Objects to the carry details for every home
    """

    def __init__(self, attributes, members):
        """
        :param attributes: dict
        a lattitude, a longitude, an index                          
        :param members: list
        the people who attend the environment 
        """

        """
        attributes: 
        :param index: int
        an identifier
        :param members: list
        a list of people who attend the environment
        :param type: string
        either 'household', 'school', or 'workplace'
        :param latitude: float
        :param longitude: float
        """
        
        
        self.index  = attributes.pop('sp_id')
        self.latitude  = attributes.pop('latitude')
        self.longitude = attributes.pop('longitude')
        self.zipcode = attributes.pop('zipcode')        
        self.__dict__.update(attributes)

        try: #assuming members are passed as records
            self.members = [member[0] for member in members]        
        except:
            self.members = members
        self.population = len(members)
        self.num_mask_types = 1
        # self.distancing = distancing
        self.total_weight = 0
        self.edges = []
        #booleans to keep track of build process
        self.hasEdges = False
        self.isWeighted = False
        #creates a dict linking each member of the environment with each prevention

        self.__dict__.update(attributes)        

    #-----------------------------------
    def drawPreventions(self, adoptions, populace):
        # populace[0]: list of properties of one person
        #print("populace: ", populace[0]); quit()
        """
        picks masks and distancing parameters
        :param adoptions: dict
        the adoptions for each prevention, with keys for environment type, to prevention type to adoption
        :param populace: dict
        the populace dict is needed to know which mask everybody is using, if necessary

        :return:
        """

        myadoptions = adoptions[self.env_type]  # self.env_type: household, school, workplace
        #----------------
        num_edges  = len(self.edges)
        num_social = int(num_edges * myadoptions["distancing"])

        #print("x num_edges= ", num_edges)
        #print("x num_social= ", num_social)

        # NOT a good approach when man businesses have only a single employee
        # But then there are no edges in the business, so there will be no effect. 
        distancing_adoption = [1] * num_social + [0] * (num_edges - num_social)
        random.shuffle(distancing_adoption)
        # class Environment
        #print(type(self.edges), type(distancing_adoption))
        #print(len(self.edges), len(distancing_adoption))
        #print("self.edges= ", self.edges)
        #print("distancing_adoption= ", distancing_adoption)
        self.distancing_adoption = dict(zip(self.edges, distancing_adoption))
        

        #assign distancers per environment
        num_distancers = int(self.population * myadoptions["distancing"])
        distancing_adoption = [1] * num_distancers + [0] * (self.population - num_distancers)
        random.shuffle(distancing_adoption)

        # number of people with masks
        num_masks = int(self.population * myadoptions["masking"])
        #print("self.population= ", self.population)
        #print("num_masks= ", num_masks)

        mask_adoption = [1] * num_masks + [0] * (self.population - num_masks)
        random.shuffle(mask_adoption)


        # mask_adoption: 0 or 1 for each member within a structured environment (workplace or school)
        self.mask_adoption = dict(zip(self.members, mask_adoption))
        # distancing_adoption: 0 or 1 for each member within a structured environment (workplace or school)



    # class Environment
    def addEdge(self, nodeA, nodeB):
        '''
        This helper function  not only makes it easier to track
        variables like the total weight and edges for the whole graph, it can be useful for debugging
        :param nodeA: int
         Index of the node for one side of the edge
        :param nodeB: int
        Index of the node for the other side
        :param weight: double
         the weight for the edge
        '''

        #self.total_weight += weight
        # NOT SURE how weight is used. 
        #self.edges.append([nodeA, nodeB, weight])
        self.edges.append((nodeA, nodeB))


    def network(self, netBuilder):
        netBuilder.buildDenseNet(self)


    def clearNet(self):
        self.edges = []
        self.total_weight = 0


class Household(Environment):
    env_type = 'household'
    def __init__(self, attributes, members):
        self.income = attributes.pop('hh_income')
        self.race = attributes.pop('hh_race')
        super().__init__(attributes, members)        


class StructuredEnvironment(Environment):
    """
    These environments are extended with a contact matrix and partition
    """

    def __init__(self, attributes, members, contact_matrix, partitioner, preventions = None):
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
        super().__init__(attributes, members)
        self.partitioner = partitioner
        self.contact_matrix = contact_matrix
        #self.total_matrix_contact = contact_matrix.sum()
        self.id_to_partition, self.partition = partitioner.partitionGroup(members)                        

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

class Workplace(StructuredEnvironment):
    env_type = 'workplace'

class School(StructuredEnvironment):
    env_type = 'school'





###############################
# define different environment types, with inheritence
################################3

