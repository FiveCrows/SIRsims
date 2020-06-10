import pandas
import numpy as np
import pickle
import PersonClass
home_folder="../Covid-19/Leon/"
filename="people.txt"
people_list={}
people_data=np.genfromtxt(home_folder+filename, skip_header=1, dtype=object)
for count,row in enumerate(people_data):
    people_list[count]=PersonClass.Person(row)
#Save data into serialized object
pickle.dump(people_list,open("people_list_serialized.pkl","wb"))

