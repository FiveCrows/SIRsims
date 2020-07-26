'''
Written by Shamik Bose. For any queries, email sb13m@my.fsu.edu
07/24/2020
This script generates contact matrices for schools and workplaces in Leon County for the following age groups:
'0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+'
APBG is calculated as the percentage of people in a particular age group compared to the total population for the original contact 
matrix. The data for the demographics is taken from the 2018 ACS tables.
The final contact matrices are written into the following files:

ContactMatrices/Leon/ContactMatrixLeonSchools.pkl
ContactMatrices/Leon/ContactMatrixLeonWorkplaces.pkl
'''
#TODO: Change APBG_workplaces to reflect splits in ages 35-44 and 65-74 according to population
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import copy
age_groups=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+']
#data from ACS corresponding to age_groups for 2018
#APBG-> Age Percentage By Groups
APBG=[0.060049726,0.060537504,0.065388298,0.065548983,0.066381795,0.071280633,0.067317127,0.065933523,0.060908112,0.063372566,0.063795481,0.06609625,0.063156716,0.052289091,0.041153316,0.06679088]
APBG_schools=[0.064503768,0.310016909,0.344630797,0.280848526,0,0,0,0,0,0,0,0,0,0,0,0,]
APBG_workplaces=[0,0,0,0.066018572,0.082834194,0.088947184,0.084001343,0.079139,0.079139,0.079343,0.079343,0.082477878,0.078809795,0.05533008, 0.05533008,0.083344668]
home_dir="RestructuredData/"
CM_home_dir="ContactMatrices/"
people_file=home_dir+"people_list_serialized_rs.pkl"
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
school_file=home_dir+"schools_list_serialized_rs.pkl"
ContactMatrixSchoolsFile=CM_home_dir+'Base/ContactMatrixUSASchools_Base.csv'
ContactMatrixWorkplacesFile=CM_home_dir+'Base/ContactMatrixUSAWorkplaces_Base.csv'
people_data=pickle.load(open(people_file,'rb'))
workplaces_data=pickle.load(open(workplaces_file,'rb'))
contactMatricesBySchoolFile=CM_home_dir+"Leon/ContactMatrixSchools.pkl"
contactMatricesByWorkplaceFile=CM_home_dir+"Leon/ContactMatrixWorkplaces.pkl"
gammaSchoolFile=CM_home_dir+"Leon/GammaSchools.pkl"
gammaWorkplaceFile=CM_home_dir+"Leon/GammaWorkplaces.pkl"
students_by_school=defaultdict(set)
employees_by_workplace=defaultdict(set)
age_groups_by_school={}
age_groups_by_workplace={}
with open(ContactMatrixSchoolsFile, 'r', encoding='utf-8-sig') as f: 
    schoolBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')  
with open(ContactMatrixWorkplacesFile, 'r', encoding='utf-8-sig') as f: 
    workplaceBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')
for row in people_data:
    record=people_data[row]
    if record.school_id:
        students_by_school[record.school_id].add((record.sp_id,record.age))
    if record.work_id:
        employees_by_workplace[record.work_id].add((record.sp_id,record.age))
def ageGroupByLoc(loc_type):
    """
    This function generates the population by age groups for loc_type (school or workplaces)
    """
    if loc_type=="school":
        dataByLoc=students_by_school
        age_group_by_loc=age_groups_by_school
    elif loc_type=="workplace":
        dataByLoc=employees_by_workplace
        age_group_by_loc=age_groups_by_workplace
    else:
        print("Unknown location type")
        return
    for key in dataByLoc.keys():
        record=dataByLoc[key]
        age_group_counts={r:0 for r in age_groups}
        for person in record:
            person_ag=person[1]//5
            if person_ag<16:
                idx=age_groups[person_ag]
            else:
                idx='75+'
            age_group_counts[idx]+=1
        age_group_by_loc[key]=age_group_counts

def buildNewGammaAndContactMatrix(loc_type):
    """
    This function builds gamma and contact matrices for loc_type (school or workplace)
    age_distribution -> percentage of population in an age group for the country
    age_groups_by_loc -> number of people in each age group, indexed by work_ or school_ id
    """
    CMNew={}
    Gammas={}
    if loc_type=="school":
        age_groups_by_loc=age_groups_by_school
        age_distribution=APBG_schools
        base_M=schoolBaseCM
        pickle_fileCM=contactMatricesBySchoolFile
        pickle_fileGamma=gammaSchoolFile
        print("Creating gamma and contact matrices for schools...")
    elif loc_type=="workplace":
        age_groups_by_loc=age_groups_by_workplace
        age_distribution=APBG_workplaces
        base_M=workplaceBaseCM
        pickle_fileCM=contactMatricesByWorkplaceFile
        pickle_fileGamma=gammaByWorkplaceFile
        print("Creating gamma and contact matrices for workplaces...")
    else:
        print("Invalid location type")
        return
    for key in age_groups_by_loc.keys():
        temp_CM=copy.deepcopy(base_M)
        temp_gamma=copy.deepcopy(base_M)
        age_counts=age_groups_by_loc[key]
        total=sum(age_counts.values())
        for age_group in age_counts:
            idx=age_groups.index(age_group)
            if age_counts[age_group]==0:
                temp_CM[idx]=0
                temp_gamma[idx]=0
            else:
                for age_groups_idx,demographic_percentage in enumerate(age_distribution):
                    if demographic_percentage:
                        temp_gamma[idx][age_groups_idx]=base_M[idx][age_groups_idx]*(1/demographic_percentage)
                        temp_CM[idx][age_groups_idx]=base_M[idx][age_groups_idx]*(1/demographic_percentage)*age_counts[age_groups[age_groups_idx]]/total
                    else:
                        temp_gamma[idx][age_groups_idx]=0
                        temp_CM[idx][age_groups_idx]=0
        CMNew[key]=temp_CM
        Gammas[key]=temp_gamma
    print("Contact matrices written to ",pickle_fileCM)
    print("Gamma values written to ",pickle_fileCM)
    pickle.dump(CMNew, open(pickle_fileCM,'wb'))
    #pickle.dump(Gammas, open(pickle_fileGamma,'wb'))

for loc_type in ["school","workplace"]:
    ageGroupByLoc(loc_type)
    buildNewGammaAndContactMatrix(loc_type)
