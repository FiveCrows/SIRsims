
#### ERROR SOMETHING MISMATCH. WHY???  
#### Output files are empty. WHY? 

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

filenm = "Results/data_baseline_p0.txt"

text = np.loadtxt(filenm)
df = pd.DataFrame(text)
df.columns = [
    "l_asymp", 
    "l_sympt", 
    "i_asymp",
    "pre_sympt",
    "i_sympt",
    "home",
    "hospital",
    "icu",
    "recov",
    "new_l_asymp",
    "new_l_sympt",
    "new_i_asympt",
    "new_pre_sympt",
    "new_i_sympt",
    "new_home",
    "new_hostp",
    "new_icu",
    "new_recov",
    "run",]  # there are 100 runs

print(df)
by = df.groupby("run")

def plot_group(by, group):
    #inf_s = []
    #pre_s = []
    #lat_s = []
    #rec_ = []
    # Different groups have different lengths
    df = by.get_group(group)
    #infected    = df["i_asymp"] + df["l_sympt"] + df["i_sympt"] + df["pre_sympt"] 
    infected    = df["i_sympt"];
    pre_sympt   = df["pre_sympt"]
    l_sympt   = df["l_sympt"]
    inf_s, = plt.plot(range(len(infected)),  infected,  color='r', label="infectious_s")
    pre_s, = plt.plot(range(len(pre_sympt)), pre_sympt, color='orange', label="pre_s")
    lat_s, = plt.plot(range(len(l_sympt)),   l_sympt,   color='b', label="latent_s")
    recov = df["recov"]
    #print("len(recov)= ", len(recov))
    rec_, = plt.plot(range(len(recov)), recov, color='g', label="recovered")
    print("-----------------\n")
    print("rec_= ", rec_, ", type= ", type(rec_))
    handles = [inf_s, pre_s, lat_s, rec_]
    labels = ['a','b','c','d']
    #handles = [inf_s[0]]
    #labels = ['a']
    return handles, labels

nb_runs = 10
for i in range(0, nb_runs):
    handles, labels = plot_group(by, i)
plt.legend(handles=handles, loc='center right', ncol=1)
plt.show()
quit()

