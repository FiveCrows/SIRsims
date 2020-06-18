'''
This program restructures the data from existing serialized files so that the key for each entry is the corresponding sp_id
'''
import gmplot
import pickle
import googlemaps
api_key="AIzaSyC14_e1FhRczF31_3kZu9lLqvtSNgGzYF8"
school_file="schools_list_serialized.pkl"
workplace_file="workplaces_list_serialized.pkl"
people_file="people_list_serialized.pkl"
households_file="households_list_serialized.pkl"
school_data=pickle.load(open(school_file, 'rb'))
workplace_data=pickle.load(open(workplace_file,'rb'))
people_data=pickle.load(open(people_file,'rb'))
households_data=pickle.load(open(households_file,'rb'))
school_data_rs, households_data_rs, workplace_data_rs, people_data_rs={},{},{},{}
# gmap = gmplot.GoogleMapPlotter(30.45, -84.25, 12) 
# lat_list,long_list=[],[]
# for row in school_data:
# 	lat=school_data[row].latitude
# 	longi=school_data[row].longitude
# 	if lat:
# 		lat_list.append(lat)
# 	if longi:
# 		long_list.append(longi)
# #print(lat_list, long_list)
# gmap.apikey='AIzaSyC14_e1FhRczF31_3kZu9lLqvtSNgGzYF8'
# gmap.scatter(lat_list, long_list, '#FF0000', size = 40, marker = False)
# gmap.polygon(lat_list, long_list,color='blue')
# gmap.draw("map1.hmtl")
res_dir="RestructuredData/"
original_data=[people_data, school_data, workplace_data, households_data]
rs_data=[people_data_rs, school_data_rs, workplace_data_rs, households_data_rs]
filenames=[people_file, school_file, workplace_file, households_file]
for idx, data in enumerate(original_data):
	for row in data:
		rs_data[idx][data[row].sp_id]=data[row]
top_keys=map(lambda x:list(x.keys())[:5],rs_data)
filenames=[people_file, school_file, workplace_file, households_file]
res_filenames=list(map(lambda x:x[:-4]+"_rs.pkl",filenames))
for idx, data in enumerate(rs_data):
	pickle.dump(data, open(res_dir+res_filenames[idx],'wb'))