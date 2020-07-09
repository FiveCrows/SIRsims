This file contains information how to run the scripts written by Shamik Bose
For any questions, email me at sb13m@my.fsu.edu

-------------------------------------------------------------------------------
augmentHSW.py
This script assigns the following dictionary to every school, workplace and household object:
self.others={'zipcode': None, 'members_count': None}
These values are then populated using the people_serialized file and the (latitude,longitude) pairs associated with the school, household and workplace files
Run: 
python augmentHSW.py
-------------------------------------------------------------------------------
augmentPeopleData.py
This script will assign comorbidity factors to the population using the health statistics given in the file 
"Morbidity and Risk Factors for American Population.txt"

Run :
python readLeonData_All.py (get unaugmented data)
python augmentPeopleData.py
-------------------------------------------------------------------------------
createTravelMatrix.py
This script creates a travel matrix for every person in the synthetic data using their school_id, work_id and sp_hh_id

Run:
python createTravelMatrix.py
-------------------------------------------------------------------------------
readCountyData.py
This script does the following:
1. Loads data from the DOH csv files (if no run parameter is specified)
2. creates plots for case counts by zipcode, county, counties above a threshold or just the top 10 counties by case count

Run:
python readCountyData.py
Other Usage:
[-d|-default]
[-county|-c] countylist (Comma separated values)
[-zipcode|-z] zipcode_list (Comma separated values)
[-threshold|-t] threshold
-------------------------------------------------------------------------------
restructureData.py
This script restructures the data from existing serialized files so that the key for each entry is the corresponding sp_id

Run:
python restructureData.py
-------------------------------------------------------------------------------
generateContactMatrix.py
This script generates contact matrices for Leon Schools in the following age groups:
3-5
6-10
11-15
16-18
The rest of the rows and columns are left unchanged from the ContactMatrixUSASchools_Base.csv file
The gamma_ii value can be changed to create a new contact matrix
gamma_ij = (1-gamma_ii)/(number of other age groups)

Run:
python generateContactMatrix.py
-------------------------------------------------------------------------------