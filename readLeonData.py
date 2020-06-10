import pandas
import numpy as np
import pickle
#Convert data from string to numbers
convertSex=lambda x: 1. if x=='F' else 0. #Sex: M=0, F=1
convertCol=lambda y: None if y.decode('UTF-8')=="X" else y.decode('UTF-8')  #Other columns: X=None
class Person:
    def __init__(self,attrs: list):
        '''
            Initialize Person objects with attributes from array
        '''
        self.sp_id=attrs[0].decode('UTF-8')
        self.sp_hh_id=attrs[1].decode('UTF-8')
        self.age=attrs[2].decode('UTF-8')
        self.sex=convertSex(attrs[3])
        self.race=attrs[4].decode('UTF-8')
        self.relate=attrs[5].decode('UTF-8')
        self.school_id=convertCol(attrs[6])
        self.work_id=convertCol(attrs[7])
home_folder="../Covid-19/Leon/"
filename="people.txt"
people_list={}
people_data=np.genfromtxt(home_folder+filename, skip_header=1, dtype=object)
for count,row in enumerate(people_data):
    people_list[row[0].decode('UTF-8')]=Person(row)
#Save data into serialized object
pickle.dump(people_list,open("people_list_serialized.pkl","wb"))

