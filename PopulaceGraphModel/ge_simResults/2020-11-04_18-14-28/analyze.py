import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import scanf
import pandas as pd


def readFile(filenm):
    fd = open(filenm, "rb")
    d = pickle.load(fd)
    ages_SIR = d['ages_SIR']
    #print("ages_SIR= ", ages_SIR)

    dct = d
    for k in dct.keys():
        print("dictionary key: ", k)
    print()
    global_dict = d['global_dict']
    for k in global_dict.keys():
        print("glob_dict key: ", k)

    """
    dct = {'sm':sm, 'sd':sd, 'wm':wm, 'wd':wd, 'red_mask': red_mask, 'red_dist': red_dist, 'v': v}
    dct['tau'] = d['params_dict']['tau']
    dct['gamma'] = d['params_dict']['gamma']
    dct['SIR'] = SIR
    dct['fn'] = filenm
    dct['preventions'] = d['preventions']
    dct['prevention_reductions'] = d['prevention_reductions']
    dct['vaccination_dict'] = d['vaccination_dict']
    print("vaccination_dict: ", dct["vaccination_dict"])
    """

    print("title: ", dct["title"])

    #dct['SIR'] = d['sim_results']  # temporary
    print("preventions: ", dct["preventions"])
    print("prevention_adoptions: ", dct["prevention_adoptions"])

    # I think that ages is the same as SIR
    """
    dct['ages'] = {}
    for k,v in ages_SIR.items():
        ages = dct['ages']
        ages[k] = v
    """

    return dct

#--------------------------------------------
dicts = []
files = glob.glob("vacci*[0-9]") 

for filenm in files:
    print("--------------------------------")
    print("filenm: ", filenm)
    dicts.append( readFile(filenm) )

df = pd.DataFrame.from_dict(dicts)
df.to_pickle("metadata.gz")
df.to_csv("metadata.csv")
#print("df['ages']= ", df['ages'][0])
#print("df['ages_SIR']= ", df['ages_SIR'][0])
#print("df['SIR']= ", df['SIR'][0])
print(df.columns)

print("-------------------")
#readFile("red_mask=0.50,red_dist=0.50,sm=0.50,sd=0.50,wm=0.50,wd=0.50,v=0.99, gamma=0.2, tau=0.2, 2020-10-19,12-37-07")
