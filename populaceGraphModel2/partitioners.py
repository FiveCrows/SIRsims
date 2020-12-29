import numpy as np 
import pandas as pd 
class Partitioner():
    def __init__(self, attribute, bin_bounds, customNames = None):
        """
        Objects of this class can be used to split a list of people into disjoint sets
        :param the name of the attribute: lambda or string
        either an attribute string or a method for getting an attribute from a populace object
        
        :param enumerator: dict
        The enumerator should map each possible values for the given attribute to the index of a partition set

        :param labels: list
        A list of names for plotting with partitioned sets
        """
        self.attribute = attribute        
        self.bin_bounds = bin_bounds
        self.num_bins = len(self.bin_bounds)-1
        self.bins = [(bin_bounds[i], bin_bounds[i+1]) for i in range(self.num_bins)]
        self.binNames = customNames or self.nameBins()

    def nameBins(self):
        names = ['{} {}'.format(self.bin_bounds[i], self.bin_bounds[i+1]) for i in range(self.num_bins)]
        return names

    def binMembers(self, members: [float]):
        raise NotImplementedError

    def partitionGroup(self, members: [np.record]):
        '''        
        :param members: list
        a list of the group members to be partitioned
        :return tuple:
        a dict from member to bin, and a dict from bin to members
        '''
        
        binList = self.binMembers(members)
        partition = {i:[] for i in range(len(self.bins))}
        for i, bin_num in enumerate(binList):
            partition[bin_num -1].append(members[i])        
        return dict(zip([member.sp_id for member in members], binList)), partition        
    

class dfPartitioner(Partitioner):
    def partitionGroup(self, members: pd.DataFrame):
        '''
        :param members: dataframe
        a list of members with attributes in a dataframe 
        '''
        pass

class directPartitioner(Partitioner):
    '''
    partition directly with an attribute
    '''
    @classmethod
    def agePartitionerA(cls):
        """
        returns a useful partitioner object for agesbins of 5 years, plus seventy five and up
        """
        bins = [i*5 for i in range(16)]
        bins.append(100)        
        return(cls('age', bins))

    def __init__(self, attribute: str, bins: list, customNames = None):
        super().__init__(attribute, bins, customNames)


    def binMembers(self, members: list):
        binList = np.digitize(list(map(lambda x: x.__getattribute__(self.attribute), members)), self.bin_bounds)
        return binList    
        

class envAttributePartitioner(Partitioner):
    pass

class customPartitioner(Partitioner):
    
    def __init__(self, attribute: str, bins: list, binNames = None, attribute_lambda = None ):
        super.__init__(self, attribute, bins, binNames)        
        self.attribute_lambda = attribute_lambda or (lambda x: x.__getattribute__(attribute))
        
    def partitionGroup(self, members: list):
        """
        This function maps each member to a partition set, and each partition sets to the subset of members who belong to it
        
        :param members: list
        A list of Person objects to partition
        
        
        :return: dict, a map from partition to 
        """
        #attribute = self.attribute
        binList = np.digitize(list(map(self.attribute_lambda, members)), self.bins)
        partition = {i:[] for i in range(len(self.bins))}
        for i, bin_num in enumerate(binList):
            partition[bin_num].append(members[i])        
        return dict(zip(members, binList)), partition
            

class enumPartitioner(Partitioner):
    '''
    depricated partitioner class
    '''
    
    def __init__(self,attribute , binBounds: list, labels=None):
        """
        :param attribute: string
        The attribute by which to partition must match one of the attributes in 'populace'

        :param enumerator: dict
        The enumerator should map each possible values for the given attribute to the index of a partition set

        :param labels: list
        A list of names for plotting with partitioned sets
        """

        self.enumerator = enumerator

        #if type(attribute) == str:            
        #    self.attribute = attribute
        

        self.labels = labels
        self.attribute_values = dict.fromkeys(set(enumerator.values()))
        self.num_sets = (len(np.unique(list(enumerator.values()))))

    def partitionGroup(self, members, populace):
        """
        This function maps each partition sets to the subset of members who belong to it

        :param members: list
        A list of indexes for the peaple to partition, if 'All' will use all members in populace
        :param populace:
        A dict associating people to a list of their attributes is required for applying the enumerator
        :return: dict
        """
        partitioned_members = {i: [] for i in range(self.num_sets)}

        for person in members:
            # determine the number for which group the person belongs in, depending on their attribute
            group = self.enumerator[populace[person][self.attribute]]
            # add person to  to dict in group
            partitioned_members[group].append(person)
        return partitioned_members

    def placeGroup(self, members, populace):
        """
        this function maps each member to the partition set they belong in
        :param members: list
        A list of indexes for the peaple to partition, if 'All' will use all members in populace
        :param populace:
        A dict associating people to a list of their attributes is required for applying the enumerator
        :return: dict
        """
        placements = {member: self.enumerator[populace[member][self.attribute]] for member in members}
        return placements

    def placeAndPartition(self, members, populace):
        """
        This function maps each partition sets to the subset of members who belong to it

        :param members: list
        A list of indexes for the peaple to partition, if 'All' will use all members in populace
        :param populace:
        A dict associating people to a list of their attributes is required for applying the enumerator
        :return: dict
        """
        partitioned_members = {i: [] for i in range(self.num_sets)}
        placements = {}
        for person in members:
            # determine the number for which group the person belongs in, depending on their attribute
            group = self.enumerator[populace[person][self.attribute]]
            # add person to  to dict in group
            partitioned_members[group].append(person)
            placements[person] = group
        return placements, partitioned_members
