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
from  partitioners import *
from synthModule import *

############################################
#load populace 
############################################
def buildPkl(slim):
    with open(currentdir+"/people_list_serialized.pkl", 'rb') as file:
        rawPopulace = pickle.load(file)
    if slim == True:        
        populace = random.choices(list(rawPopulace.values()), k = int(0.1*len(rawPopulace)))
        print("WARNING! slim = True, 90% of people are filtered out")
    else:
        populace = list(rawPopulace.values())
    
    #switch object over so pickle will load using synthModule
    #as a list of objects    
    populace_df = pd.DataFrame.from_records(pop.__dict__ for pop in populace)
    #floats to ints
    populace_df.work_id = populace_df.work_id.fillna(0).astype(int).replace(0,np.nan)
    populace = populace_df.to_records()
    
    population = len(populace)
    # print("self.populace: ", self.populace); quit()
    print("population: ", population)

    # To sort people into a dict of categories.
    # For example, if one wants the indexes of all people with the some school id, they could do
    #    pops_by_category['school_id'][someInt]
    # takes a dict of dicts to represent populace and returns a list of dicts of
    #    lists to represent groups of people with the same attributes
    #  Give an example fo person[category], and an example of category
    attributes = list(populace_df.keys())
    
    attributes = attributes[:-1]
    pops_by_category = {category: {} for category in attributes}
    #how to sort by df, slow
    #pops_by_category = {category: populace_df.groupby(category)['sp_id'].apply(list) for category in attributes}
    quickPopList = [(vars(rawPopulace[key])) for key in rawPopulace] 
    print("ok, here") # .transpose()
    for index in range(len(populace)):
        person = quickPopList[index]
        populace[index].sp_id = index
        for category in attributes:
            try:
                pops_by_category[category][person[category]].append(index)
            except:
                pops_by_category[category][person[category]] = [index]
    
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

    for index in raw_hh:
        hh = raw_hh[index]
        index = hh.sp_id
        #get member records from pops_by_category indexes            
        try:
            
            members =  pops_by_category['sp_hh_id'][hh.sp_id]                    
        except:
            members = []            

        envDict = hh.__dict__
        envDict['zipcode'] = envDict.pop('others')['zipcode']
        environments[index] = Household(envDict, members)

    #turn populace into dataframe

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
        print(index)
        try:
            members = [populace[i] for i in pops_by_category['work_id'][index]]
        except:
            members = []
        envDict = wp.__dict__
        envDict['zipcode'] = envDict.pop('others')['zipcode']        
        environments[index] = Workplace(envDict, members, contact_matrices[index], partitioner)
    
        

    #build schools
    raw_s = pickle.load(open(currentdir + "/schools_list_serialized.pkl", 'rb'))
    with open(currentdir + "/ContactMatrixSchools.pkl", 'rb') as file:
        contact_matrices = pickle.load(file)
    for index in raw_s:
        s = raw_s[index]
        index = s.sp_id        
        print(index)                
        
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


    with open(fName, 'wb') as file:
        pickle.dump(output,file)
        print('dumped to ' + fName)

buildPkl(False)
buildPkl(True)
