Module LoadData

readLeonData

end

import numpy as np
import seaborn, os, sys
from datetime import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
home_dir='zipcode_data/'
convert_nonstring=lambda x: 0 if not x.isnumeric() else x
file_list=os.listdir(home_dir)
county_data_file="county_data_file_combined.pkl"
county_data_file_dates="county_data_file_dates.pkl"
def readData():
    '''
        Read data from pickle file if it exists and reads newer data if available
        If not, reads from .csv files and puts them into pickle file
        Returns list of dates for which data is available
    '''
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
    return (dates,data)

def structureData(dates: list, data):
    sorted_dates=list(sorted(dates))
    counties=set()
    for row in data[sorted_dates[0]].iterrows():
        counties.add(row[1].COUNTYNAME)
    county_list=list(counties)
    cumulativeData=np.zeros((len(counties),len(sorted_dates)))
    #Put data into numpy array for easy manipulation and access
    for c, date in enumerate(sorted_dates):
        for row in data[date].iterrows():
            x_idx=county_list.index(row[1].COUNTYNAME)
            cumulativeData[x_idx][c]+=float(row[1].Cases_1)
    #Uncomment following block to print county-wise cumulative data
    # for idx, r in enumerate(cumulativeData):
    #     print(county_list[idx])
    #     for c in r:
    #         print(c, end=" ")
    #     print()
    return (cumulativeData, county_list)
#Find counties with more than a threshold no. of cases

def main(threshold: int):
    '''
        To run this program, you will need files as given below:
        home_dir='zipcode_data/' - This will be the folder containing all the csv files
        county_data_file="county_data_file_combined.pkl" - Contains older data. If unavailable, it will be generated. If newer data is present, this will be updated
        county_data_file_dates="county_data_file_dates.pkl" - Contains list of dates. If unavailable, it will be generated. If newer data is present, this will be updated
    '''
    dates,data=readData()
    cumulativeData,county_list=structureData(dates, data)
    threshold=threshold
    hr_counties=[]
    for idx, r in enumerate(cumulativeData):
        if r[-1]>=threshold:
            hr_counties.append(idx)
    for v in hr_counties:
        plt.plot(sorted(dates), cumulativeData[v], label=county_list[v])
    plt.xticks(rotation=45)
    plt.legend()
    title_temp="Counties with cumulative case count over "+ str(threshold)
    plt.title(title_temp)
    fig_suffix=datetime.now().strftime("%d%m%y%H%M")
    #print(fig_suffix)
    plt.show()
    plt.savefig("CasesGraph"+fig_suffix)

if __name__=="__main__":
    if len(sys.argv)==1:
        print("Enter threshold for plotting graphs")
    elif sys.argv[1].isnumeric():
        main(int(sys.argv[1]))
