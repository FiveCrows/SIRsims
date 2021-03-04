"""
The partitioner classes for this module are used to split lists of people
for creating partitioned environments, analyzing, and plotting 
into groups by records of their attributes 

direct partitioner is useful to partition by age, for example, which is conveniently available as the class method 
agePartitionerA

autoBinLambdaPartitioner is written to group people into bins of uniform size. Is an effective partitioner 

"""
import numpy as np 
import pandas as pd
import math 
import time
class Partition():
    def __init__(self, sets, labels, attribute):    
        """
        Partition objects describe how people can be 
        seperated into sets.         
        :param sets: a dict of partition sets
        :param labels: labels to describe the partition sets
        :param attribute: a name for the attribute by which the sets are partitioned        
        """        

        self.attribute = attribute
        self.sets = sets        
        self.num_sets = len(sets)
        self.labels = labels
        self.placements = None

    def subPartition(self, members):
        self.subPartition(self, members)
    def get_placements(self):
        """
        placements is the inverse dict.
        given an individual, it maps their set.
        """
        placements = self.placements or self.invert_dict(self.sets)
        self.placements = placements
        return placements

    def invert_dict(self, dict):
        """for get placements
        seperated so the 'or' can be used
        """
        inverted_dict = {}
        for k,v in self.sets.items():
            for v_i in v: 
                inverted_dict[v_i] = k
        return inverted_dict        

class OrdinalPartition(Partition):
    """
    a partition by groups of numbers, 
    so sets can be named by number ranges
    """
    def __init__(self, sets, bounds, attribute):
        self.bounds = bounds
        super().__init__(sets, self.labelBins(), attribute)

    def labelBins(self):
        n = len(self.bounds)
        text = []
        for bound in self.bounds:
            if abs(bound)>1000 and abs(bound) !=np.inf:
                text.append('{}k'.format(round(bound/1000)))
            else:
                text.append('{}'.format(bound))
        delim = '-'
        labels = [text[i]+delim+text[i+1] for i in range(n-1)]
        return labels
        
            
                
            

        return names



class Partitioner():
    def __init__(self, bounds, customNames = None):
        """
        Objects of this class can be used to split a list of people into disjoint sets
        :param the name of the attribute: lambda or string
        either an attribute string or a method for getting an attribute from a populace object
        
        :param enumerator: dict
        The enumerator should map each possible values for the given attribute to the index of a partition set

        :param bounds: list
        This list should include each boundary between bins,
        but the first and last bound are implicity at inf
        :param labels: list
        A list of names for plotting with partitioned sets
        """        

        self.bounds = bounds
        self.num_bins = len(self.bounds)-1
        self.bins = [(bounds[i], bounds[i+1]) for i in range(self.num_bins)]        
        self.binNames = customNames or self.nameBins()
        
    def nameBins(self):
        names = ['{}:{}'.format(self.bounds[i], self.bounds[i+1]) for i in range(self.num_bins-2)]
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
        partition = {i:[] for i in range(len(self.bounds)-1)}
        for i, bin_num in enumerate(binList):
            #add each member to the partition by id
            partition[bin_num].append(members[i][0])
        bins = dict(zip([member.sp_id for member in members], binList)), partition        
        return(Partition(bins, self.binNames, self.binBounds))


class directPartitioner(Partitioner):
    '''
    partition directly with an attribute
    '''
    @classmethod
    def agePartitionerA(cls):
        """
        returns a useS partitioner object for agesbins of 5 years, plus seventy five and up
        """
        bins = [i*5 for i in range(16)]
        bins.append(100)        
        return(cls('age', bins))

    def __init__(self, attribute: str, bins: list, customNames = None):
        self.attribute = attribute        
        super().__init__(bins, customNames)


    def binMembers(self, members: list):

        binList = np.digitize(list(map(lambda x: x.__getattribute__(self.attribute), members)), self.bounds)
        binList = [i-1 for i in binList] #so that the bin numbering starts with zero
        return binList    


class autoBinLambdaPartitioner(Partitioner):    
    """
    This partioner is built with partitioning by income in mind.     
    incomePartitioner = autoBinLambdaPartitioner(lambda pers: model.environments[pers.sp_hh_id].income, 5)
    since the bounds are autogenerated, they are stored but will change each time a partitioning is done. 
    :param f: function
    :param num_bins: int

    """
    @classmethod
    def incomePartitioner(cls, households, num_bins):
        """
        partitioning with the income partitioner will place people into groups with an even number of households
        :param households: list
        a list of households to define a lambda which relates a person to their household
        :param num_bins: int
        """
        #to group evenly people into some number of bins by their income
        return cls(lambda pers: households[pers.sp_hh_id].income, num_bins, 'income')

    def __init__(self, f, num_bins: int, attribute = None):
        self.f = f
        self.num_bins = num_bins
        self.num_sets = num_bins
        self.attribute = attribute
    def partitionGroupWithAutoBound(self, members: list):
        #it's simpler than it looks. it just sorts and splits the numpy array evenly
        
        #another idea as to how to sort... 
        #f_pairs = [(member,self.f(member)) for member in members]
        #f_pairs.sort(key = lambda x: x[0])
        #members, incomes = list(zip(*f_pairs))        
        members.sort(key = self.f)
        member_id = {members[i].sp_id:i for i in range(len(members))}
        sets = [list(x) for x in np.array_split(list(member_id.keys()), self.num_bins)]        
        #let the bounds be the first of each bin, and also the last of the last bin 
        bounds = [self.f(members[member_id[partition[0]]]) for partition in sets]
        bounds = [-math.inf]+ bounds[1:] + [math.inf]
        super(autoBinLambdaPartitioner, self).__init__(bounds)                
        return(OrdinalPartition(dict(enumerate(sets)), bounds, self.attribute))
        
    def binMembers(self, members: list):
        binList = np.digitize(list(map(self.f, members)), self.bounds)   
        binList = [i-1 for i in binList]
        return binList
        

class customPartitioner(Partitioner):
    
    def __init__(self, attribute: str, bins: list, binNames = None, attribute_lambda = None ):
        super.__init__(self, attribute, bins, binNames)        
        self.attribute_lambda = attribute_lambda or (lambda x: x.__getattribute__(attribute))
        
    def partitionGroup(self, members: list):
        """
        This function maps each member to a partition set, and each partition sets to the subset of members who belong to it
        
        :param members: list
        A list of Person Objects to partition
        
        
        :return: dict, a map from partition to 
        """
        #attribute = self.attribute
        binList = np.digitize(list(map(self.attribute_lambda, members)), self.bins)
        #start indexing from 0 instead of 1
        binList = [i-1 for i in binList]
        partition = {i:[] for i in range(len(self.bins_bounds+1))}

        for i, bin_num in enumerate(binList):
            partition[bin_num].append(members[i])        
        sets =  dict(zip(members, binList)), partition
        return(OrdinalPartition(sets, self.binNames, self.binBounds))
            

class nominalPartitioner(Partitioner):
    """
    can be used for partitioning nominal, unordered sets
    """
    
    
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
