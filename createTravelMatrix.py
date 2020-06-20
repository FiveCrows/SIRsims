'''
This script creates a travel matrix for every person in the synthetic data
'''

import pickle
from travel import Travel

home_dir="RestructuredData/"
school_file=home_dir+"schools_list_serialized_rs.pkl"
workplace_file=home_dir+"workplaces_list_serialized_rs.pkl"
people_file=home_dir+"people_list_serialized_rs.pkl"
households_file=home_dir+"households_list_serialized_rs.pkl"
school_data=pickle.load(open(school_file, 'rb'))
workplace_data=pickle.load(open(workplace_file,'rb'))
people_data=pickle.load(open(people_file,'rb'))
households_data=pickle.load(open(households_file,'rb'))
travel_matrix={} #Dictionary indexed by person's sp_id. People who go to work OR school will have four entries, people who do both will have 6
schoolAndWorkCount=0 #This data keeps track of all people who go to school AND work. For Leon, that number is 1469
studentCount=0
employedCount=0
unemployedCount_F, unemployedCount_M=0,0
oldCount,infantCount=0,0
for person in people_data:
	record=people_data[person]
	home=households_data[record.sp_hh_id]
	home_loc=(home.latitude,home.longitude)	
	workplace_loc, school_loc=None, None
	if record.school_id and record.work_id:
		schoolAndWorkCount+=1
		workplace=workplace_data[record.work_id]
		workplace_loc=(workplace.latitude,workplace.longitude)
		school=school_data[record.school_id]
		school_loc=(school.latitude,school.longitude)
	elif record.school_id:
		studentCount+=1
		school=school_data[record.school_id]
		school_loc=(school.latitude,school.longitude)
	elif record.work_id:
		employedCount+=1
		workplace=workplace_data[record.work_id]
		workplace_loc=(workplace.latitude,workplace.longitude)
	elif 65>record.age>18:
		if record.sex==0:
			unemployedCount_M+=1
		else:
			unemployedCount_F+=1
	if record.age<4:
		infantCount+=1
	elif record.age>65:
		oldCount+=1
	travel_matrix[record.sp_id]=Travel([home_loc, workplace_loc, school_loc])
# print("Student and Employed: ",schoolAndWorkCount,"\nStudent Only: ",studentCount,"\nEmployed Only: ", employedCount, "\nUnemployed: ", unemployedCount)
# print(infantCount, oldCount)
# print("Male unemployed: ", unemployedCount_M, "\nFemale unemployed: ",unemployedCount_F )
pickle.dump(travel_matrix, open("travel_matrix_file",'wb'))
