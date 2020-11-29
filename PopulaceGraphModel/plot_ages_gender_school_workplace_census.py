import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
from unicodedata import normalize



enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)-1) for i in range(15)]
names.append("{}:{}".format(75,100))

df = pd.read_csv("ages_gender_school_workplace.csv")

df_population = pd.read_csv("../datasets/2010 population.csv", skiprows=1)
df_population.set_index('Age', inplace=True)

df2 = pd.read_csv("../datasets/Leon_EMP_ACSST5Y2018.S2301_data_with_overlays_2020-11-20T105515.csv", skiprows=1)
df2 = df2.append((pd.read_csv("../datasets/US_EMP_ACSST5Y2018.S2301_data_with_overlays_2020-11-20T105132.csv", skiprows=1)))
# print(df2.head())

fieldlist = []
fieldlist.append ('Geographic Area Name')
fieldlist.append('Estimate!!Total!!Population 16 years and over')
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!16 to 19 years') # 4 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!20 to 24 years') # 5 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!25 to 29 years') # 5 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!30 to 34 years') # 5 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!35 to 44 years') # 10 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!45 to 54 years') # 10 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!55 to 59 years') # 5 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!60 to 64 years') # 5 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!65 to 74 years') # 10 years
                            # Total!!Estimate!!AGE!!65 to 74 years
fieldlist.append('Estimate!!Total!!Population 16 years and over!!AGE!!75 years and over') #25 years
df2018_workplace=df2[fieldlist]

fieldlist = ['Location','Total','15:19','20:24']
fieldlist +=['25:29','30:34','35:44','45:54', '55:59','60:64','65:74','75:100']
df2018_workplace.columns = fieldlist

df2018_workplace['35:39'] = df2018_workplace['35:44']/2
df2018_workplace['40:44'] = df2018_workplace['35:44']/2
df2018_workplace['45:49'] = df2018_workplace['45:54']/2
df2018_workplace['50:54'] = df2018_workplace['45:54']/2
df2018_workplace['65:69'] = df2018_workplace['65:74']/2
df2018_workplace['70:74'] = df2018_workplace['65:74']/2
df2018_workplace['Year'] = '2018'
df2018_workplace['0:4']= 0
df2018_workplace['5:9']= 0
df2018_workplace['10:14']= 0

# print(df2018_workplace)
# index = pd.MultiIndex.from_tuples([('United States','Year')])
# cols = [['wrk'],['United States','Leon County, Florida']]
# rows = names

df2 = pd.read_csv("../datasets/Leon_EMP_ACSST5Y2010.S2301_data_with_overlays_2020-11-20T105515.csv", skiprows=1)
df2 = df2.append((pd.read_csv("../datasets/US_EMP_ACSST5Y2010.S2301_data_with_overlays_2020-11-20T105132.csv", skiprows=1)))

fieldlist = []
fieldlist.append ('Geographic Area Name')
fieldlist.append('Total!!Estimate!!Population 16 years and over')
fieldlist.append('Total!!Estimate!!AGE!!16 to 19 years') # 4 years
fieldlist.append('Total!!Estimate!!AGE!!20 to 24 years') # 5 years
fieldlist.append('Total!!Estimate!!AGE!!25 to 44 years') # 20 years
fieldlist.append('Total!!Estimate!!AGE!!45 to 54 years') # 10 years.
fieldlist.append('Total!!Estimate!!AGE!!55 to 64 years') # 10 years.
fieldlist.append('Total!!Estimate!!AGE!!65 to 74 years') # 5 years
fieldlist.append('Total!!Estimate!!AGE!!75 years and over') # 25 years
# print(fieldlist)



df2010_workplace=df2[fieldlist]
fieldlist =  ['Location','Total' , '15:19', '20:24','25:44']
fieldlist += ['45:54', '55:64', '65:74', '75:100']
df2010_workplace.columns = fieldlist


df2010_workplace['25:29'] = df2010_workplace['25:44']/4
df2010_workplace['30:34'] = df2010_workplace['25:44']/4
df2010_workplace['35:39'] = df2010_workplace['25:44']/4
df2010_workplace['40:44'] = df2010_workplace['25:44']/4
df2010_workplace['45:49'] = df2010_workplace['45:54']/2
df2010_workplace['50:54'] = df2010_workplace['45:54']/2
df2010_workplace['55:59'] = df2010_workplace['55:64']/2
df2010_workplace['60:64'] = df2010_workplace['55:64']/2
df2010_workplace['65:69'] = df2010_workplace['65:74']/2
df2010_workplace['70:74'] = df2010_workplace['65:74']/2

df2010_workplace['Year'] = '2010'
df2010_workplace['0:4']= 0
df2010_workplace['5:9']= 0
df2010_workplace['10:14']= 0
# print(df2010_workplace)



locations = ['United States', 'Leon County, Florida']
years = ['2010','2018']
multi_levels = pd.MultiIndex.from_product([locations,years], names = ['Location', 'Year'])
df_workplace = pd.concat([df2010_workplace,df2018_workplace])
df_workplace = df_workplace.set_index(multi_levels)
df_workplace = df_workplace[names]
df_workplace = df_workplace.transpose()
for i,c in enumerate(df_workplace.columns):
    df_workplace[c] = df_workplace[c] / sum(df_workplace[c])

# df_workplace.reset_index()
# print(multi_levels)
# print(df_workplace)


df2 = pd.read_csv("../datasets/SCHOOL_ACSST5Y2018.S1401-2020-11-28T210044.csv")
df2['Label'] = df2['Label'].str.strip()
df2['Label'] = df2['Label'].str.normalize('NFKD')

fieldlist =  ['Label','United States Total', 'United States Percent', 'United States In public school']
fieldlist += ['United States Percent in public school', 'United States In private school','United States Percent in private school']
fieldlist += ['Leon County Florida Total', 'Leon County Florida Percent', 'Leon County Florida In public school']
fieldlist += ['Leon County Florida Percent in public school', 'Leon County, Florida In private school']
fieldlist += ['Leon County Florida Percent in private school']
df2.columns= fieldlist



df3 = pd.read_csv("../datasets/School_ACSST5Y2010.S1401-2020-11-28T210824.csv")
df3['Label'] = df3['Label'].str.strip()

fieldlist =  ['Label', 'United States Total', 'United States Percent in public school','United States Percent in private school']
fieldlist += ['Leon County Florida Total', 'Leon County Florida In public school', 'Leon County Florida Percent in private school']
df3.columns = fieldlist

df2 = df2[fieldlist] 
df2 = df2.append(pd.DataFrame([['Year'] + ([2018] * 6)], columns = fieldlist), ignore_index=True)
df2 = df2.append(pd.DataFrame([['Location'] + (['United States'] * 3) +(['Leon County Florida'] * 3)], columns = fieldlist), ignore_index=True)


df3 = df3.append(pd.DataFrame([['Year'] + ([2010] * 6)], columns = fieldlist), ignore_index=True)
df3 = df3.append(pd.DataFrame([['Location'] + (['United States'] * 3) +(['Leon County Florida'] * 3)], columns = fieldlist), ignore_index=True)
print(df3)

df2=df2.applymap(lambda x: x.replace(',', '') if (isinstance(x, str)) else x)
df3=df3.applymap(lambda x: x.replace(',', '') if (isinstance(x, str)) else x)




# df2 = df.set_index('Label')
df2.set_index(df2.columns[0], inplace = True)
df3.set_index(df3.columns[0], inplace = True)
df2=df2.transpose()
df3=df3.transpose()

df2.to_csv('try2018.csv')
df3.to_csv('try2010.csv')

fieldlist =  ['Location', 'Year', 'Population 3 to 4 years', '3 to 4 year olds enrolled in school', 'Population 5 to 9 years']
fieldlist += ['5 to 9 year olds enrolled in school', 'Population 10 to 14 years', '10 to 14 year olds enrolled in school']
fieldlist += ['Population 15 to 17', '15 to 17 year olds enrolled in school', 'Population 18 to 19 years', '18 and 19 year olds enrolled in school']

df2= df2 [fieldlist].loc[ ['United States Total' , 'Leon County Florida Total'] , : ]


# loc[df['column_name'] == 'United States Total']
df_School =pd.DataFrame()
df_School['Location'] = df2['Location']
df_School['Year'] = df2['Year']
df_School['0:4']=   df2['3 to 4 year olds enrolled in school']
df_School['5:9']=   df2['5 to 9 year olds enrolled in school']
df_School['10:14']= df2['10 to 14 year olds enrolled in school']
df_School['15:19']= df2['15 to 17 year olds enrolled in school'].astype(int) + df2['18 and 19 year olds enrolled in school'].astype(int)
df_School['20:24'] = 0
df_School['25:29'] = 0
df_School['30:34'] = 0
df_School['35:39'] = 0
df_School['40:44'] = 0
df_School['45:49'] = 0
df_School['50:54'] = 0
df_School['55:59'] = 0
df_School['60:64'] = 0
df_School['65:69'] = 0
df_School['70:74'] = 0
df_School['75:100'] =0


# ['Location','Total' , '15:19', '20:24','25:44']
# fieldlist += ['45:54', '55:64', '65:74', '75:100']
print(df_School)


# now I have to convert pct population to total population for US, will need to get it for Leon
fieldlist =['Location', 'Year', '3 and 4 years', '5 to 9 years', '10 to 14 years', '15 to 17 years','18 and 19 years']
df3 = df3[fieldlist].loc[ ['United States Total' , 'Leon County Florida Total'] , : ]
# print(df3)
df_School3 = pd.DataFrame()
df_School3['Location'] = df3['Location']
df_School3['Year'] = df3['Year']



# Please check this next section for me:
# 0:4 is not same as 3 to 4
df_School3['0:4']=   df3['3 and 4 years'].str.rstrip('%').astype('float')/100 * [df_population.loc['0:4','US Population'],df_population.loc['0:4','Leon Population']]
df_School3['5:9']=   df3['5 to 9 years'].str.rstrip('%').astype('float')/100 * [df_population.loc['5:9','US Population'],df_population.loc['5:9','Leon Population']]
df_School3['10:14']= df3['10 to 14 years'].str.rstrip('%').astype('float')/100 * [df_population.loc['10:14','US Population'],df_population.loc['10:14','Leon Population']]
df_School3['15:19']= (df3['15 to 17 years'].str.rstrip('%').astype('float')+ df3['18 and 19 years'].str.rstrip('%').astype('float'))/100 * [df_population.loc['15:19','US Population'],df_population.loc['15:19','Leon Population']]
df_School3['20:24'] = 0
df_School3['25:29'] = 0
df_School3['30:34'] = 0
df_School3['35:39'] = 0
df_School3['40:44'] = 0
df_School3['45:49'] = 0
df_School3['50:54'] = 0
df_School3['55:59'] = 0
df_School3['60:64'] = 0
df_School3['65:69'] = 0
df_School3['70:74'] = 0
df_School3['75:100'] =0



df_School = df_School.append(df_School3)
df_School = df_School.transpose()
print(df_School)

# df = pd.DataFrame({'ages':xxxxx, 'age_sch':ages_school, 'age_wrk':ages_workplace, 'm_sch':m_sch, 'f_sch':f_sch, 'm_wrk':m_wrk, 'f_wrk':f_wrk})

# I want to distinguish subplot according to school/work. So I should create a dataframe with two columns
#  ages, nb, gender, 




#-----------------------------------------------------
# Census 2010
# https://www.census.gov/prod/cen2010/briefs/c2010br-03.pdf
M=[10319427,10389638,10579862,11303666,11014176,10635591,
9996500,10042022,10393977,11209085,10933274,
9523648,8077500,5852547,4243972,3182388,2294374,1273867,
424387,82263,9162]
F=[9881935, 9959019,10097332,10736677,10571823,10466258,
9965599,10137620,10496987,11499506,11364851,10141157,
8740424,6582716,5034194,4135407,3448953,2346592,1023979,
288981,44202]

# Age groups: 0-4, ..., 70-74, 75-79,80-84,85,89,90-94,95-100, 100-over
# Combine last 7 slots into 1
MM = M[:-6]
MM.append(sum(M[-7:]))    #### SOMETHING WRONG WITH DIMENSIONS. But if correct, I get 15/16 dichotomy. 
FF = F[:-6]
FF.append(sum(F[-7:]))
US_M, US_F = MM, FF
# print(len(MM), len(FF))
# print(names)
# print("df.shape: ", df.shape)
# print("US_M: ", len(US_M))

df['US_M'] = US_M
df['US_F'] = US_F

# For all the columns, divide by the sum over the column
# print(df.columns)
for i,c in enumerate(df.columns):
    if i <= 1: continue
    df[c] = df[c] / sum(df[c])

#-----------------------------------------------------
cols = [['wrk','wrk','sch','sch','US','US'],['m','f','m','f','m','f']]
rows = names
# print (df)
data = np.asarray(list(map(list, [df['m_wrk'].values,df['f_wrk'].values,df['m_sch'].values,df['f_sch'].values, df['US_M'].values, df['US_F'].values]))).T
# print(data.shape)

#print("rows= ", rows)
#print("cols= ", cols)
#print('data= ', data)
df = pd.DataFrame(data, index=rows, columns=cols)
#-----------------------------------------------------

df1 = df.stack()
df1 = df1.stack()
df1.index.set_names(['ages','sex','env'], inplace=True)
df1 = pd.DataFrame(df1)
df1.columns = ['value']
df1.reset_index(inplace=True)
# print("df1= ", df1)

# How to do arithmetic on DF columns

#sns.set(rc={'figure.figsize':(11,8.57)})
g = sns.FacetGrid(df1, col='env') #, height=4.5) #, aspect=11.5/8.5)
mm = g.map(sns.barplot,'ages', 'value', 'sex', lw=0, palette="Blues")
mm.add_legend()
mm.fig.suptitle("Population Age Distribution in the schools+workplaces", fontsize=14)
g.set_xticklabels(rotation=45, fontsize=6)
#plt.title("Population Age Distribution in the schools+workplaces", fontsize=14)
plt.tight_layout()
plt.savefig("plot_demographics.jpg")
plt.savefig("plot_demographics.pdf")

quit()

# Demographics in the US
# https://www.infoplease.com/us/census/demographic-statistics

"""
Male
138,053,563
49.1
Female
143,368,343
50.9
 
 
 
Under 5 years
19,175,798
6.8
5 to 9 years
20,549,505
7.3
10 to 14 years
20,528,072
7.3
15 to 19 years
20,219,890
7.2
20 to 24 years
18,964,001
6.7
25 to 34 years
39,891,724
14.2
35 to 44 years
45,148,527
16.0
45 to 54 years
37,677,952
13.4
55 to 59 years
13,469,237
4.8
60 to 64 years
10,805,447
3.8
65 to 74 years
18,390,986
6.5
75 to 84 years
12,361,180
4.4
85 years and over
4,239,587
1.5
"""

# Census 2010
# https://www.census.gov/prod/cen2010/briefs/c2010br-03.pdf
M=[10319427,10389638,10579862,11303666,11014176,10635591,
9996500,10042022,10393977,11209085,10933274,
9523648,8077500,5852547,4243972,3182388,2294374,1273867,
424387,82263,9162]
F=[9881935, 9959019,10097332,10736677,10571823,10466258,
9965599,10137620,10496987,11499506,11364851,10141157,
8740424,6582716,5034194,4135407,3448953,2346592,1023979,
288981,44202]

# Age groups: 0-4, ..., 70-74, 75-79,80-84,85,89,90-94,95-100, 100-over
# Combine last 7 slots into 1
MM = M[:-7]
MM.append(sum(M[-7:]))
FF = F[:-7]
FF.append(sum(F[-7:]))
M, F = MM, FF
print(len(M))

# Integrate M, N, into the dataframe: M_US, F_US
print(df1)
df1.to_csv('ages_gender_school_workplace.csv')

plt.savefig("plot_demographics.jpg")


