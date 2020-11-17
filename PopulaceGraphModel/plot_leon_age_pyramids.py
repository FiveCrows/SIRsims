import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ge_modelingToolkit2 import *
import numpy as np

save_output = False
slim = False

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)-1) for i in range(15)]
names.append("{}:{}".format(75,100))
partitioner = Partitioner('age', enumerator, names)

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(14)]
names.append("75:100")
partitioner = Partitioner('age', enumerator, names)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}  
# denotes the fraction of people using masks
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}

sim=False
model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim, timestamp=timestamp)

h_sizes = [0]*13
s_sizes = [0]*35
w_sizes = [0]*10

envs = {'household':[], 'school':[], 'workplace':[]}
for index in model.environments:
    env = model.environments[index]
    envs[env.env_type].extend(env.members)

print("schools: ", len(envs['school']))
print("workplaces: ", len(envs['workplace']))
print("homes: ", len(envs['household']))

def computeAgeHist(members, populace):
    ages = [0]*16
    # Compute age histogram in specified environment
    for ix in members:
        ages[enumerator[populace[ix]['age']]] += 1 # not most efficient
    return ages

def computeAgeGenderHist(members, populace):
    ages_male   = [0]*16
    ages_female = [0]*16
    # Compute age histogram in specified environment
    for ix in members:
        p = populace[ix]
        if p['sex'] == 0:  # female
            ages_male[enumerator[populace[ix]['age']]] += 1 # not most efficient
        else:
            ages_female[enumerator[populace[ix]['age']]] += 1 # not most efficient
    return ages_male, ages_female

#------------------------------------------
ages_school = computeAgeHist(envs['school'], model.populace)
ages_workplace = computeAgeHist(envs['workplace'], model.populace)
m_sch, f_sch = computeAgeGenderHist(envs['school'], model.populace)
m_wrk, f_wrk = computeAgeGenderHist(envs['workplace'], model.populace)
print("ages_schools: ", ages_school)
print("ages_workplace: ", ages_workplace)
print("ages(M,F)_schools: ", list(zip(m_sch, f_sch)))
print("ages(M,F)_workplace: ", list(zip(m_wrk, f_wrk)))

#df = pd.DataFrame(names, ages_school, ages_workplace, 
pop = model.population
for i in range(pop):
    print(model.populace[0])
quit()

"""
    environment = model.environments[environment]
    if environment.env_type == 'household':
        h_sizes[environment.population]+=1
    if environment.env_type == 'school':
        s_sizes[environment.population//100] +=1
    if environment.env_type == 'workplace':
        w_sizes[int(np.log(environment.population))] +=1
"""


df = pd.DataFrame({\
       'Age': ['0-4','5-9','10-14','15-19','20-24','25-29','30-34',\
               '35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74',
               '75-79','80-84','85-89','90-94','95-99','100+'], 
       'Male': [-49228000, -61283000, -64391000, -52437000, -42955000, -44667000, -31570000, \
                -23887000, -22390000, -20971000, -17685000, -15450000, -13932000, -11020000, \
                -7611000, -4653000, -1952000, -625000, -116000, -14000, -1000], \
       'Female': [52367000, 64959000, 67161000, 55388000, 45448000, 47129000, 33436000, 26710000, \
                  25627000, 23612000, 20075000, 16368000, 14220000, 10125000, 5984000, 3131000, \
                  1151000, 312000, 49000, 4000, 0]})


AgeClass = ['100+','95-99','90-94','85-89','80-84','75-79','70-74','65-69','60-64','55-59','50-54',
            '45-49','40-44','35-39','30-34','25-29','20-24','15-19','10-14','5-9','0-4']

bar_plot = sns.barplot(x='Male', y='Age', data=df, order=AgeClass, lw=0)
bar_plot = sns.barplot(x='Female', y='Age', data=df, order=AgeClass, lw=0)

plt.savefig("pop_pyramid.jpg")
