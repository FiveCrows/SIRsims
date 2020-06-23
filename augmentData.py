'''
This script will assign comorbidity factors to the population using the health statistics given in the file 
"Morbidity and Risk Factors for American Population.txt"
IMPORTANT:
Run readLeonData_All.py before running this to have unaugmented data
'''
import pickle
import random
from collections import defaultdict
people_file="people_list_serialized.pkl"
people_data=pickle.load(open(people_file, 'rb'))
people_list=list(people_data.keys())
age_groups=['2-19','20-44','45-64','65+','others']
sex_groups=['M','F']
age_groupings=defaultdict(list)
sex_groupings=defaultdict(list)
for idx in people_list:
    person=people_data[idx]
    if  2<=person.age<=19:
        age_groupings['2-19'].append(idx)
    elif 20<=person.age<=44:
        age_groupings['20-44'].append(idx)
    elif 45<=person.age<=64:
        age_groupings['45-64'].append(idx)
    elif person.age>=65:
        age_groupings['65+'].append(idx)
    else:
        age_groupings['others'].append(idx)
    if person.sex==0:
        sex_groupings['M'].append(idx)
    else:
        sex_groupings['F'].append(idx)

#Hypertension
ht_groups=['2-19M','2-19F','20-44M','20-44F','45-64M','45-64F','65+M','65+F','othersM','othersF']
ht_percentages=[31.3, 28.7,73.1,37.9, 50.1, 42,1,51.7, 55.8, 31.3, 28.7]
#The following loop creates groupings for ht_groups above
for idx, group in enumerate(ht_groups):
    age_set=set(age_groupings[group[:-1]])
    sex_set=set(sex_groupings[group[-1]])
    common=age_set&sex_set
    samples=random.sample(common, int(ht_percentages[idx]*len(common)/100))
    for sample in samples:
        people_data[sample].comorbidities['Hypertension']=True
#Diabetes
diabetes_groups=['2-19','20-44','45-64','65+','others']
diabetes_percentages=[0, 5.6, 21.9, 28.2, 0]
#The following loop creates groupings for diabetes_groups above
for idx, group in enumerate(diabetes_groups):
    samples=random.sample(age_groupings[group], int(diabetes_percentages[idx]*len(age_groupings[group])/100))
    for sample in samples:
        people_data[sample].comorbidities['Diabetes']=True
#Obesity
obesity_groups=['2-19M','2-19F','20-44M','20-44F','45-64M','45-64F','65+M','65+F','othersM','othersF']
obesity_percentages=[19.1,17.8,38.1,41.2,38.1,41.2,38.1,41.2,0,0]
#The following loop creates groupings for ht_groups above
for idx, group in enumerate(obesity_groups):
    age_set=set(age_groupings[group[:-1]])
    sex_set=set(sex_groupings[group[-1]])
    common=age_set&sex_set
    samples=random.sample(common, int(obesity_percentages[idx]*len(common)/100))
    for sample in samples:
        people_data[sample].comorbidities['Obesity']=True
#heart_disease
hd_groups=['2-19M','2-19F','20-44M','20-44F','45-64M','45-64F','65+M','65+F','othersM','othersF']
hd_percentages=[2.5, 2.5, 3.6, 4.3, 13.2, 11.0, 35.4, 24.5, 0,0]
#The following loop creates groupings for ht_groups above
for idx, group in enumerate(hd_groups):
    age_set=set(age_groupings[group[:-1]])
    sex_set=set(sex_groupings[group[-1]])
    common=age_set&sex_set
    samples=random.sample(common, int(hd_percentages[idx]*len(common)/100))
    for sample in samples:
        people_data[sample].comorbidities['Heart disease']=True

#Lung Disease
ld_groups=['2-19M','2-19F','20-44M','20-44F','45-64M','45-64F','65+M','65+F','othersM','othersF']
ld_percentages=[22.2, 18.7, 9.0, 14.1, 15.9, 23.8, 16.0, 23.0, 0,0]
#The following loop creates groupings for ht_groups above
for idx, group in enumerate(ld_groups):
    age_set=set(age_groupings[group[:-1]])
    sex_set=set(sex_groupings[group[-1]])
    common=age_set&sex_set
    samples=random.sample(common, int(ld_percentages[idx]*len(common)/100))
    for sample in samples:
        people_data[sample].comorbidities['Lung disease']=True

hd_count, ht_count, obese_count, ld_count, db_count=0,0,0,0,0
for idx in people_list:
    person=people_data[idx]
    if person.comorbidities['Heart disease']:
        hd_count+=1
    if person.comorbidities['Hypertension']:
        ht_count+=1
    if person.comorbidities['Obesity']:
        obese_count+=1
    if person.comorbidities['Lung disease']:
        ld_count+=1
    if person.comorbidities['Diabetes']:
        db_count+=1
print("Hypertensive:", ht_count)
print("Heart Disease:", hd_count)
print("Obese:", obese_count)
print("Diabetic:", db_count)
print("Lung Disease:", ld_count)
pickle.dump(people_data, open(people_file,'wb'))
print("Augmented data written into people_list_serialized.pkl")