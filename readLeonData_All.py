import pandas
import numpy as np
import pickle
import os
import SynthData
home_folder="../Covid-19/Leon/"
#Get data for people
people_file="people.txt"
people_list={}
people_data=np.genfromtxt(home_folder+people_file, skip_header=1, dtype=object)
for count,row in enumerate(people_data):
    people_list[count]=SynthData.Person(row)
#Save data into serialized object
pickle.dump(people_list,open("people_list_serialized.pkl","wb"))
#Get data for households
households_file="households.txt"
households_list={}
households_data=np.genfromtxt(home_folder+households_file, skip_header=1, dtype=object)
for count,row in enumerate(households_data):
    households_list[count]=SynthData.Household(row)
#Save household data into serialized object
pickle.dump(households_list,open("households_list_serialized.pkl","wb"))
#Get data for workplaces
workplaces_file="workplaces.txt"
workplaces_list={}
workplaces_data=np.genfromtxt(home_folder+workplaces_file, skip_header=1, dtype=object)
for count,row in enumerate(workplaces_data):
    workplaces_list[count]=SynthData.Workplace(row)
#Save household data into serialized object
pickle.dump(workplaces_list,open("workplaces_list_serialized.pkl","wb"))
#Get data for schools
schools_file="schools.txt"
schools_list={}
schools_data=np.genfromtxt(home_folder+schools_file, skip_header=1, dtype=object)
for count,row in enumerate(schools_data):
    schools_list[count]=SynthData.School(row)
#Save household data into serialized object
pickle.dump(schools_list,open("schools_list_serialized.pkl","wb"))
