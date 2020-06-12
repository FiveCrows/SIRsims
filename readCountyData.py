#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import seaborn, os
from datetime import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
home_dir='zipcode_data/'
convert_nonstring=lambda x: 0 if not x.isnumeric() else x
file_list=os.listdir(home_dir)
county_data_file="county_data_file_combined.pkl"
county_data_file_dates="county_data_file_dates.pkl"
if os.path.isfile(county_data_file_dates):
    dates=pickle.load(open(county_data_file_dates,'rb'))
    data=pickle.load(open(county_data_file,'rb'))
    print("Earlier data loaded. Latest date is ",datetime.strptime(str(sorted(dates)[-1]),'%m%d%Y'))
else:
    dates=set()
    data={} #key is date, value is a dictionary
flag=0
for filename in file_list:
    file_date=filename[-12:-4]
    if filename[-3:]!='csv':
        continue
    elif file_date in dates:
        continue
    else:
        flag=1
        dates.add(file_date)
        print("Reading from ",home_dir+filename)
        data_temp=pd.read_csv(home_dir+filename,header=0,usecols=['ZIP','OBJECTID_1','COUNTYNAME','Cases_1'], converters={'Cases_1':convert_nonstring})
        data[file_date]=data_temp

if flag==0:
    print("No newer data available")
if flag==1:
    print("New data loaded till ", sorted(dates)[-1])
    pickle.dump(dates,open(county_data_file_dates,'wb'))
    pickle.dump(data,open(county_data_file,'wb'))


# In[113]:


# sorted_dates=list(sorted(dates))
# counties=["Suwannee", "Leon", "Volusia", "Duval"]
# zipcode=32712
# y_vals=[[],[],[],[]]
#print(len(data[query_date]))
# for date in sorted_dates:
#     date_sum=[0,0,0,0]
#     #print(date)
#     for row in data[date].iterrows():
#         if row[1].iloc[2]==counties[0]:
#             date_sum[0]+=int(row[1].iloc[3])
#         elif row[1].iloc[2]==counties[1]:
#             date_sum[1]+=int(row[1].iloc[3])
#         elif row[1].iloc[2]==counties[2]:
#             date_sum[2]+=int(row[1].iloc[3])
#         elif row[1].iloc[2]==counties[3]:
#             date_sum[2]+=int(row[1].iloc[3])
#     y_vals[0].append(date_sum[0])
#     y_vals[1].append(date_sum[1])
#     y_vals[2].append(date_sum[2])
#     y_vals[3].append(date_sum[3])
# plt.plot(sorted_dates, y_vals[0])
# plt.plot(sorted_dates, y_vals[1])
# plt.plot(sorted_dates, y_vals[2])
# plt.plot(sorted_dates, y_vals[3])
# plt.xticks(rotation=45)
# plt.suptitle(county)
# plt.show()


# In[ ]:




