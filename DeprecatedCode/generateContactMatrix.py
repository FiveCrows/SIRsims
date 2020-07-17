'''
Written by Shamik Bose. For any queries, email sb13m@my.fsu.edu
07/08/2020
This script generates contact matrices for Leon Schools in the following age groups:
3-5
6-10
11-15
16-18
The rest of the rows and columns are left unchanged from the ContactMatrixUSASchools_Base.csv file
The gamma_ii value can be changed to create a new contact matrix
gamma_ij = (1-gamma_ii)/(number of other age groups)
'''
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import copy
home_dir="RestructuredData/"
people_file=home_dir+"people_list_serialized_rs.pkl"
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
school_file=home_dir+"schools_list_serialized_rs.pkl"
households_file=home_dir+"households_list_serialized_rs.pkl"
people_data=pickle.load(open(people_file,'rb'))
workplaces_data=pickle.load(open(workplaces_file,'rb'))
gamma_ii=0.7 #This is the probability of contacting someone from the same age group
students_by_school=defaultdict(set)
min_age_overall, max_age_overall=50,0
for row in people_data:
    record=people_data[row]
    if record.school_id:
        students_by_school[record.school_id].add((record.sp_id,record.age))
        if record.age<min_age_overall:
            min_age_overall=record.age
        elif record.age>max_age_overall:
            max_age_overall=record.age
#print(min_age,max_age) min_age is 3, max_age is 18
age_counts_by_school=defaultdict(dict)
for key in students_by_school.keys():
    age_counts={r: 0 for r in range(min_age_overall,max_age_overall+1)}
    school_distr=students_by_school[key]
    #Set counts per age
    for student_records in school_distr:
        age_counts[student_records[1]]+=1
    #assign counts per age group as in Contact Matrix
    age_group_counts={'3-4':sum([age_counts[r] for r in range(min_age_overall,5)]),'5-9':sum([age_counts[r] for r in range(5,10)]),
                     '10-14':sum([age_counts[r] for r in range(10,15)]),'15-18':sum([age_counts[r] for r in range(15,max_age_overall+1)])}
    #Remove age groups that aren't present
    for age in ['3-4','5-9','10-14','15-18']:
        if age_group_counts[age]==0:
            del(age_group_counts[age])
    age_counts_by_school[key]=age_group_counts
with open('ContactMatrixUSASchools_Base.csv', 'r', encoding='utf-8-sig') as f: 
    schoolBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')
#Not currently being used  
with open('ContactMatrixUSAWorkplaces_Base.csv', 'r', encoding='utf-8-sig') as f: 
    workplaceBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')
with open('ContactMatrixUSA_Base.csv', 'r', encoding='utf-8-sig') as f: 
    BaseCM= np.genfromtxt(f, dtype=float, delimiter=',')
contact_matrices={}
age_group_map={'3-4':0,'5-9':1,'10-14':2,'15-18':3}
for key in age_counts_by_school.keys():
    gamma_ij=0 #This is the probability of contacting someone from a different age group
    CMbySchool=copy.deepcopy(schoolBaseCM)
    age_counts=age_counts_by_school[key]
    total=sum(age_counts.values())
    
    #Ensuring schools that have only one age group are not omitted
    if len(age_counts)==1:
        contact_matrices[key]=CMbySchool
        continue
    else:
        gamma_ij=(1-gamma_ii)/(len(age_counts)-1)
    for age_group in age_counts:
        remaining=age_counts.keys()-{age_group}
        CMbySchool[age_group_map[age_group]][age_group_map[age_group]]*=gamma_ii*age_counts[age_group]/total
        for ag in remaining:
            CMbySchool[age_group_map[age_group]][age_group_map[ag]]*=gamma_ij*age_counts[ag]/total
    contact_matrices[key]=CMbySchool
#Writing contact matrix back into pickled file for easy reading
pickle.dump(contact_matrices, open("ContactMatrixLeonSchools.pkl",'wb'))

print(type(contact_matrices))
