# Author: Bryan Azbill
# This script constructs pickle files to load environments and populace easily, slim/not slim
from collections import defaultdict
import dill
import pickle
import os, sys
import pandas as pd


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

sys.path.append(parentdir)
from modelingToolkit import *
from partitioners import *
from synthModule import *

############################################
#load populace 
############################################
def buildPkl(slim):
    with open(currentdir+"/people_list_serialized.pkl", 'rb') as fd:
        rawPopulace = pickle.load(fd)
    if slim == True:        
        # GE: fixed error: 0.9 -> 0.1
        populace = random.choices(list(rawPopulace.values()), k = int(0.1*len(rawPopulace)))
        print("WARNING! slim = True, 90% of people are filtered out")
    else:
        populace = list(rawPopulace.values())
    
    # populace is a list of dictionaries. Each dictionary has attributes (race, gender, etc)
    populace_df = pd.DataFrame.from_records(pop.__dict__ for pop in populace)
    populace_df.work_id = populace_df.work_id.fillna(-1.)
    populace_df.school_id = populace_df.school_id.fillna(-1.)
    populace = populace_df.to_records()
    populace_df.to_csv("populace.csv")
    population = len(populace)

    print("population: ", population)

    # To sort people into a dict of categories.
    # For example, if one wants the indexes of all people with the some school id, they could do
    #    pops_by_category['school_id'][someInt]
    # takes a dict of dicts to represent populace and returns a list of dicts of
    #    lists to represent groups of people with the same attributes
    #  Give an example fo person[category], and an example of category
    attributes = list(populace_df.keys())
    attributes.remove('comorbidities')

    pops_by_category = {category: {} for category in attributes}
    pops_by_category = {category: defaultdict(list) for category in attributes}
    #how to sort by df, slow
    #pops_by_category = {category: populace_df.groupby(category)['sp_id'].apply(list) for category in attributes}
    quickPopList = [(vars(rawPopulace[key])) for key in rawPopulace] 
    print("ok, here") # .transpose()
    for index in range(len(populace)):
        person = quickPopList[index]
        for category in attributes:
            pops_by_category[category][person[category]].append(index)
    
    #for people as objects
    #for person in populace:
    #    for category in attributes:
    #        try:
    #            pops_by_category[category][person[category]].append(person.sp_id)
    #        except:
    #            pops_by_category[category][person[category]] = [person.sp_id]
    

    #########################################
    # Build and Load environments into dict
    ########################################
    print("checkpoint achieved")
    environments = defaultdict(None)
    #load households
    raw_hh = pickle.load(open(currentdir + "/households_list_serialized.pkl", 'rb'))
    #print(raw_hh.keys()) # list of integers
    #print(raw_hh[1])
    #print(type(raw_hh[1]))
    # 'hh_income', 'hh_race', 'latitude', 'longitude', 'others', 'sp_id', 'stcotrbg']
    # Access via raw_hh[1].hh_income
    #print(dir(raw_hh[1]))

    for index in raw_hh:
        hh = raw_hh[index]
        index = hh.sp_id
        members =  pops_by_category['sp_hh_id'][hh.sp_id]                    

        envDict = vars(hh)
        # envDict['others'] is a dictionary: {'zipcode': '32303', 'members_count': 2})
        # Using pop returns the value popped. Therefore, 'zipcode' can be accessed. Slick. 
        envDict['zipcode'] = envDict.pop('others')['zipcode']
        environments[index] = Household(envDict, members)

    #create a partitioner    
    partitioner = directPartitioner.agePartitionerA()

    #load workplaces
    raw_wp = pickle.load(open(currentdir + "/workplaces_list_serialized.pkl", 'rb'))

    #build workplaces
    with open(currentdir + "/ContactMatrixWorkplaces.pkl", 'rb') as file:
        contact_matrices = pickle.load(file)

    for index in raw_wp:
        wp = raw_wp[index]        
        index = wp.sp_id
        try:
            members = [populace[i] for i in pops_by_category['work_id'][index]]
        except:
            members = []
        envDict = vars(wp)
        envDict['zipcode'] = envDict.pop('others')['zipcode']        
        environments[index] = Workplace(envDict, members, contact_matrices[index], partitioner)
    
    #build schools
    raw_s = pickle.load(open(currentdir + "/schools_list_serialized.pkl", 'rb'))
    with open(currentdir + "/ContactMatrixSchools.pkl", 'rb') as file:
        contact_matrices = pickle.load(file)
    for index in raw_s:
        s = raw_s[index]
        index = s.sp_id        
        
        try:
            members = [populace[i] for i in pops_by_category['school_id'][index]]            
        except:
            members = []
        envDict = s.__dict__
        envDict['zipcode'] = envDict.pop('others')['zipcode']
        environments[index] = School(envDict, members, contact_matrices[index], partitioner)
        
    # because the zipcode loads as a string
    for environment in environments.values():
        try:
            environment.zipcode = int(environment.zipcode)
        except:
            environment.zipcode = None
    

    #store file
    if slim == False:
        fName = currentdir+"/leon.pkl"
    elif slim == True:
        fName = currentdir+"/slimmedLeon.pkl"

    #split into two parts because files don't fit together
    output = {
            'populace': populace,
            'pops_by_category': pops_by_category,    
            'environments': environments,
            'partitioner': partitioner,
            }

    # list of attributes, but with no keys. Dangerous. How do you know what is what? 
    print("populace[0:3]: ", populace[0:3])  
    # pops_by_category:  dict_keys(['sp_id', 'sp_hh_id', 'age', 'sex', 'race', 'relate', 'school_id', 'work_id'])
    #print("pops_by_category.keys(): ", pops_by_category.keys())
    # sp_id is the 'id' of the school, workplace, or home
    #print("pops_by_category['sp_id'].keys()= ", pops_by_category['sp_id'].keys())
    print("pops_by_category['age'][80]= ", pops_by_category['age'][80])
    print("len(environments): ", len(environments))
    env_keys = list(environments.keys())
  
    # Produces an object of type Household, Workplace, or School
    for i in range(len(env_keys)):
        env = environments[env_keys[i]]
        if env.env_type == 'household':
            print("HOUSEHOLD")
            print(dir(env))
            print(type(env))
            break
    for i in range(len(env_keys)):
        env = environments[env_keys[i]]
        if env.env_type == 'school':
            print("SCHOOL")
            print(dir(env))
            print(type(env))
            print("partition= ", env.partition)
            print("partitioner= ", env.partitioner)
            print("population= ", env.population)
            break
    for i in range(len(env_keys)):
        env = environments[env_keys[i]]
        if env.env_type == 'workplace':
            print("WORKPLACE")
            print(dir(env))
            print(type(env))
            print("partition= ", env.partition)
            print("partitioner= ", env.partitioner)
            print("population= ", env.population)
            break
    print("partitioner: ", dir(partitioner))
#partitioner:  ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'agePartitionerA', 'attribute', 'binMembers', 'binNames', 'bin_bounds', 'bins', 'nameBins', 'num_bins', 'partitionGroup']
    print("partitioner.bin_bounds: ", partitioner.bin_bounds)
    print("partitioner.binNames: ", partitioner.binNames)
    print("partitioner.nameBins: ", partitioner.nameBins)
    print("partitioner.num_bins: ", partitioner.num_bins)
    print("partitioner.partitionGroup: ", partitioner.partitionGroup)

    with open(fName, 'wb') as file:
        pickle.dump(output,file)
        print('dumped to ' + fName)

#buildPkl(False)
buildPkl(True)
