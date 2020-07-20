'''
Written by Shamik Bose. For any queries, email sb13m@my.fsu.edu
07/17/2020
This script generates contact matrices for schools and workplaces in Leon County for the following age groups:
'0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+'
The gamma_ii value can be changed to create a new contact matrix (set to 0.7 for schools, 0.5 for workplaces)
gamma_ij = (1-gamma_ii)/(number of other age groups present)
Contact from present to absent age groups are set to zero
The final contact marices are written into the following files:

ContactMatrixLeonSchools.pkl
ContactMatrixLeonWorkplaces
'''
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import copy
age_groups=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+']
home_dir="RestructuredData/"
people_file=home_dir+"people_list_serialized_rs.pkl"
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
school_file=home_dir+"schools_list_serialized_rs.pkl"
ContactMatrixSchoolsFile='ContactMatrixUSASchools_Base.csv'
ContactMatrixWorkplacesFile='ContactMatrixUSAWorkplaces_Base.csv'
people_data=pickle.load(open(people_file,'rb'))
workplaces_data=pickle.load(open(workplaces_file,'rb'))
contactMatricesBySchoolFile="ContactMatrixLeonSchools.pkl"
contactMatricesByWorkplaceFile="ContactMatrixLeonWorkplaces.pkl"
with open(ContactMatrixSchoolsFile, 'r', encoding='utf-8-sig') as f: 
    schoolBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')  
with open(ContactMatrixWorkplacesFile, 'r', encoding='utf-8-sig') as f: 
    workplaceBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')
workplaceContactMatrices={}
schoolContactMatrices={}
school_gamma_ii=0.7 #This is the probability of contacting someone from the same age group in school
workplace_gamma_ii=0.5 #This is the probability of contacting someone from the same age group in the workplace
students_by_school=defaultdict(set)
employees_by_workplace=defaultdict(set)
#Get data by workplace and school
for row in people_data:
    record=people_data[row]
    if record.school_id:
        students_by_school[record.school_id].add((record.sp_id,record.age))
    if record.work_id:
        employees_by_workplace[record.work_id].add((record.sp_id,record.age))
#Put students into age_groups
age_groups_by_school={}
for key in students_by_school.keys():
    schoolRecord=students_by_school[key]
    age_group_counts={r:0 for r in age_groups}
    #print(age_group_counts)
    for student in schoolRecord:
        student_ag=student[1]//5
        idx=age_groups[student_ag]
        #print(student_ag,idx)
        age_group_counts[idx]+=1
    for age_group in age_groups: #dropping absent age groups
        if age_group_counts[age_group]==0:
            del(age_group_counts[age_group])
    age_groups_by_school[key]=age_group_counts
#Put employees into age groups
age_groups_by_workplace={}
for key in employees_by_workplace.keys():
    workplaceRecord=employees_by_workplace[key]
    age_group_counts={r:0 for r in age_groups}
    for employee in workplaceRecord:
        employee_ag=employee[1]//5
        if employee_ag<16:
            idx=age_groups[employee_ag]
        else:
            idx='75+'
        age_group_counts[idx]+=1
    for age_group in age_groups: #dropping absent age groups
        if age_group_counts[age_group]==0:
            del(age_group_counts[age_group])
    age_groups_by_workplace[key]=age_group_counts
#The following loop creates contact matrices by school
contactMatricesBySchool={}
for key in age_groups_by_school.keys():
    temp_gamma_ii=school_gamma_ii
    gamma_ij=0 #This is the probability of contacting someone from a different age group
    CMbySchool=copy.deepcopy(schoolBaseCM)
    age_counts=age_groups_by_school[key]
    total=sum(age_counts.values())
    absentAgeGroups=set(age_groups)-age_counts.keys()
    #Ensuring schools that have only one age group are not omitted
    if len(age_counts)==1:
        temp_gamma_ii=1
    else:
        gamma_ij=(1-temp_gamma_ii)/(len(age_counts)-1)
    for age_group in age_counts:
        remaining=age_counts.keys()-{age_group}
        CMbySchool[age_groups.index(age_group)][age_groups.index(age_group)]*=temp_gamma_ii*age_counts[age_group]/total
        for ag in remaining:
            CMbySchool[age_groups.index(age_group)][age_groups.index(ag)]*=gamma_ij*age_counts[ag]/total
        for ag in absentAgeGroups: #All absent age_groups set to zero
            CMbySchool[age_groups.index(age_group)][age_groups.index(ag)]=0
    contactMatricesBySchool[key]=CMbySchool
pickle.dump(contactMatricesBySchool, open(contactMatricesBySchoolFile,'wb'))

#The following loop creates contact matrices by workplace
contactMatricesByWorkplace={}
for key in age_groups_by_workplace.keys():
    temp_gamma_ii=workplace_gamma_ii
    gamma_ij=0 #This is the probability of contacting someone from a different age group
    CMbyWorkplace=copy.deepcopy(workplaceBaseCM)
    age_counts=age_groups_by_workplace[key]
    total=sum(age_counts.values())
    absentAgeGroups=set(age_groups)-age_counts.keys()
    if len(age_counts)==1:
        temp_gamma_ii=1
    else:
        gamma_ij=(1-temp_gamma_ii)/(len(age_counts)-1)
    for age_group in age_counts:
        remaining=age_counts.keys()-{age_group}
        CMbyWorkplace[age_groups.index(age_group)][age_groups.index(age_group)]*=temp_gamma_ii*age_counts[age_group]/total
        for ag in remaining:
            CMbyWorkplace[age_groups.index(age_group)][age_groups.index(ag)]*=gamma_ij*age_counts[ag]/total
        for ag in absentAgeGroups: #All absent age_groups set to zero
            CMbyWorkplace[age_groups.index(age_group)][age_groups.index(ag)]=0
    contactMatricesByWorkplace[key]=CMbyWorkplace
pickle.dump(contactMatricesByWorkplace, open(contactMatricesByWorkplaceFile,'wb'))