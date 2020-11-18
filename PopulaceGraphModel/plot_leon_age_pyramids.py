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
        if p['sex'] == 1:  # female
            ages_male[enumerator[populace[ix]['age']]] += 1 # not most efficient
        else:
            ages_female[enumerator[populace[ix]['age']]] += 1 # not most efficient
    return ages_male, ages_female

#------------------------------------------
ages_school = computeAgeHist(envs['school'], model.populace)
ages_workplace = computeAgeHist(envs['workplace'], model.populace)
m_sch, f_sch = computeAgeGenderHist(envs['school'], model.populace)
m_wrk, f_wrk = computeAgeGenderHist(envs['workplace'], model.populace)

print()
print("ages_schools: ", ages_school)
print("ages_workplace: ", ages_workplace)
print("ages_f_wrk: ", f_wrk)
print("ages_m_wrk: ", m_wrk)
print("ages_f_sch: ", f_sch)
print("ages_m_sch: ", m_sch)

print()
print("ages(M,F)_schools: ", list(zip(m_sch, f_sch)))
print("ages(M,F)_workplace: ", list(zip(m_wrk, f_wrk)))
print(names)
print(len(names))
print(len(m_wrk))
print(enumerator)

df = pd.DataFrame({'ages':names, 'age_sch':ages_school, 'age_wrk':ages_workplace, 'm_sch':m_sch, 'f_sch':f_sch, 'm_wrk':m_wrk, 'f_wrk':f_wrk})
df.to_csv("ages_gender_school_workplace.csv")
quit()
#----------------------------------------------------------------------
