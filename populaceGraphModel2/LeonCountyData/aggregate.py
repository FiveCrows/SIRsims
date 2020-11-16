#this script constructs pickles to load environments and populace easily, slim/not slim
import pickle
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

sys.path.append(parentdir)
from modelingToolkit import *

############################################
#load populace#
############################################
def buildPkl(slim):
    with open(currentdir+"/people_list_serialized.pkl", 'rb') as file:
        rawPopulace = pickle.load(file)
    if slim == True:
        print("WARNING! slim = True, 90% of people are filtered out")
        populace = []
        for key in rawPopulace:
            if random.random() > 0.9:
                populace.append(vars(rawPopulace[key]))
    else:
        populace = [(vars(rawPopulace[key])) for key in rawPopulace]  # .transpose()
    population = len(populace)
    # print("self.populace: ", self.populace); quit()
    print("population: ", population)

    # To sort people into a dict of categories.
    # For example, if one wants the indexes of all people with the some school id, they could do
    #    pops_by_category['school_id'][someInt]
    # takes a dict of dicts to represent populace and returns a list of dicts of
    #    lists to represent groups of people with the same attributes
    #  Give an example fo person[category], and an example of category

    # count total nb of nodes

    # env_name_alternate = {"household": "sp_hh_id", "work": "work_id", "school": "school_id"} outdated


    environments = {}
    attributes = list(populace[0].keys())
    attributes = attributes[:-1]
    pops_by_category = {category: {} for category in attributes}

    for index in range(len(populace)):
        person = populace[index]
        for category in attributes:
            try:
                pops_by_category[category][person[category]].append(index)
            except:
                pops_by_category[category][person[category]] = [index]

    pops_by_category["age_groups"] = {}




    for bracket in range(0, 20):
        pops_by_category["age_groups"][bracket] = []
        for i in range(0, 5):
            try:  # easier than conditionals. I divided all ages into groups of 5
                pops_by_category["age_groups"][bracket].extend(pops_by_category["age"][5 * bracket + i])
            except:
                continue

    raw_hh = pickle.load(open(currentdir + "/households_list_serialized.pkl", 'rb'))
    for index in raw_hh:
        hh = raw_hh[index]
        dict = hh.__dict__
        dict['index'] = dict.pop('sp_id')
        dict['zipcode'] = dict.pop('others')['zipcode']
        try:
            env = Household(dict, pops_by_category['sp_hh_id'][(dict['index'])])
        except:
            env = Household(dict, [])
        environments[index] = env


    #create a partitioner
    enumerator = {i:i//5 for i in range(75)}
    enumerator.update({i:15 for i in range(75,100)})
    names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
    partitioner = Partitioner('age', enumerator, names)

    raw_wp = pickle.load(open(currentdir + "/workplaces_list_serialized.pkl", 'rb'))
    #build workplaces
    with open(currentdir + "/ContactMatrixWorkplaces.pkl", 'rb') as file:
        contact_matrices = pickle.load(file)
    for index in raw_wp:
        wp = raw_wp[index]
        dict = wp.__dict__
        dict['index'] = dict.pop('sp_id')
        index = dict['index']
        dict['zipcode'] = dict.pop('others')['zipcode']
        print(index)
        try:
            workplace = Workplace(dict, pops_by_category['work_id'][index], populace, contact_matrices[index],
                                  partitioner)
        except:
            workplace = Workplace(dict, [], populace, contact_matrices[index],
                                  partitioner)
        environments[index] = workplace

    #build schools
    raw_s = pickle.load(open(currentdir + "/schools_list_serialized.pkl", 'rb'))
    with open(currentdir + "/ContactMatrixSchools.pkl", 'rb') as file:
        contact_matrices = pickle.load(file)
    for index in raw_s:
        s = raw_s[index]
        dict = s.__dict__
        print(index)
        dict['index'] = dict.pop('sp_id')
        index = dict['index']
        dict['zipcode'] = dict.pop('others')['zipcode']
        try:
            school = School(dict, pops_by_category['school_id'][index], populace, contact_matrices[index], partitioner)
        except:
            school = School(dict, [], populace, contact_matrices[index], partitioner)
        environments[index] = (school)

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
    else:
        print(" slim bool?? type it right,  scrub!")

    output= {'partitioner': partitioner,
             'populace': populace,
             'pops_by_category': pops_by_category,
             'partitioner': partitioner,
             'environments': environments}

    with open(fName, 'wb') as file:
        pickle.dump(output,file)
        print('dumped to ' + fName)

buildPkl(True)
buildPkl(False)