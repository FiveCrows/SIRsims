'''
It also assigns the following dictionary to every school, workplace and household object:
self.others={'zipcode': None, 'members_count': None}
These values are then populated using the people_serialized file and the latitude,longitude pairs associated with 
'''
from arcgis.gis import GIS
from arcgis.geocoding import geocode, reverse_geocode
from arcgis.geometry import Point
from collections import defaultdict
import pickle
import synthdata
from tqdm import tqdm
gis=GIS()
home_dir="RestructuredData/"
people_file=home_dir+"people_list_serialized_rs.pkl"
people_data=pickle.load(open(people_file,'rb'))
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
workplace_data=pickle.load(open(workplaces_file,'rb'))
school_file=home_dir+"schools_list_serialized_rs.pkl"
school_data=pickle.load(open(school_file,'rb'))
households_file=home_dir+"households_list_serialized_rs.pkl"
households_data=pickle.load(open(households_file,'rb'))
data_in=[workplace_data, school_data, households_data]
files=[workplaces_file, school_file, households_file]
#The following two loops maps lat,long to a zipcode
for data in data_in:
	for row in tqdm(data):
	    info=data[row]
	    location={
	        'Y':info.latitude,
	        'X': info.longitude
	    }
	    pt=Point(location)
	    address=reverse_geocode(location=pt)
	    zipcode=address['address']['Postal']
	    data[row].others['zipcode']=zipcode
#The following loop increments member count for each school and workplace
print("Updating member counts for households, workplaces and schools")
for row in tqdm(people_data):
    person=people_data[row]
    if person.school_id:
        school_data[person.school_id].others['members_count']+=1
    if person.work_id:
        workplace_data[person.work_id].others['members_count']+=1
    if person.sp_hh_id:
    	households_data[person.sp_hh_id].others['members_count']+=1
#Writing augmented data back to files
for idx in range(len(data_in)):
	pickle.dump(data_in[idx], open(files[idx],'wb'))
	print("Data written to ",files[idx])




