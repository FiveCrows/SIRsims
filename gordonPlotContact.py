'''
Written by Shamik Bose. For any queries, email sb13m@my.fsu.edu
07/24/2020
This script generates contact matrices for schools and workplaces in Leon County for the following age groups:
'0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+'
APBG is calculated as the percentage of people in a particular age group compared to the total population for the original contact 
matrix. The data for the demographics is taken from the 2018 ACS tables.
The final contact matrices are written into the following files:

ContactMatrices/Leon/ContactMatrixLeonSchools.pkl
ContactMatrices/Leon/ContactMatrixLeonWorkplaces.pkl

# PLEASE PROVIDE LINKS TO DATA SOURCE(S)
'''
#TODO: Change APBG_workplaces to reflect splits in ages 35-44 and 65-74 according to population
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import copy
age_groups=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75+']
#data from ACS corresponding to age_groups for 2018
#APBG-> Age Percentage By Groups
APBG=[0.060049726,0.060537504,0.065388298,0.065548983,0.066381795,0.071280633,0.067317127,0.065933523,0.060908112,0.063372566,0.063795481,0.06609625,0.063156716,0.052289091,0.041153316,0.06679088]
APBG_schools=[0.064503768,0.310016909,0.344630797,0.280848526,0,0,0,0,0,0,0,0,0,0,0,0,]
APBG_workplaces=[0,0,0,0.066018572,0.082834194,0.088947184,0.084001343,0.079139,0.079139,0.079343,0.079343,0.082477878,0.078809795,0.05533008, 0.05533008,0.083344668]
home_dir="RestructuredData/"
CM_home_dir="ContactMatrices/"
people_file=home_dir+"people_list_serialized_rs.pkl"
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
school_file=home_dir+"schools_list_serialized_rs.pkl"
ContactMatrixSchoolsFile=CM_home_dir+'Base/ContactMatrixUSASchools_Base.csv'
ContactMatrixWorkplacesFile=CM_home_dir+'Base/ContactMatrixUSAWorkplaces_Base.csv'
people_data=pickle.load(open(people_file,'rb'))
workplaces_data=pickle.load(open(workplaces_file,'rb'))
contactMatricesBySchoolFile=CM_home_dir+"Leon/ContactMatrixSchools.pkl"
contactMatricesByWorkplaceFile=CM_home_dir+"Leon/ContactMatrixWorkplaces.pkl"
gammaSchoolFile=CM_home_dir+"Leon/GammaSchools.pkl"
gammaWorkplaceFile=CM_home_dir+"Leon/GammaWorkplaces.pkl"
students_by_school=defaultdict(set)
employees_by_workplace=defaultdict(set)
age_groups_by_school={}
age_groups_by_workplace={}
with open(ContactMatrixSchoolsFile, 'r', encoding='utf-8-sig') as f: 
    schoolBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')  
with open(ContactMatrixWorkplacesFile, 'r', encoding='utf-8-sig') as f: 
    workplaceBaseCM= np.genfromtxt(f, dtype=float, delimiter=',')


print(schoolBaseCM)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=16)
matplotlib.rc('xtick', labelsize=12) # axis tick labels
matplotlib.rc('ytick', labelsize=12) # axis tick labels
matplotlib.rc('axes', labelsize=12)  # axis label
matplotlib.rc('axes', titlesize=12)  # subplot title
matplotlib.rc('figure', titlesize=12)

plt.subplots(1,2)

plt.suptitle("Contact Matrices\nSchools (left), Workplaces (right)", fontsize=16)
plt.subplot(1,2,1)
plt.imshow(schoolBaseCM, origin='lower')
plt.xlabel("age bracket")
plt.ylabel("age bracket")

plt.subplot(1,2,2)
plt.imshow(workplaceBaseCM, origin='lower')
plt.xlabel("age bracket")
plt.ylabel("age bracket")
plt.tight_layout()

plt.tight_layout()
plt.savefig("plot_contact_matrices.pdf")
