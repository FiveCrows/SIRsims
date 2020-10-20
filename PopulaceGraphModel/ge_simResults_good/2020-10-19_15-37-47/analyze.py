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
    SIR = d['sim_results']
    print("title: ", d['title'])
    print("d.keys(): ", d.keys())
    print("preventions: ", d['preventions'])
    print("preventions_reductions: ", d['prevention_reductions'])
    print("params_dict: ", d['params_dict'])
    print("params_dict: ", d['params_dict'])

    red_mask, red_dist, sm, sd, wm, wd, v = scanf.scanf("red_mask=%f,red_dist=%f,sm=%f,sd=%f,wm=%f,wd=%f,v=%f", d['title'])
    dct = {'sm':sm, 'sd':sd, 'wm':wm, 'wd':wd, 'red_mask': red_mask, 'red_dist': red_dist, 'v': v}
    dct['tau'] = d['params_dict']['tau']
    dct['gamma'] = d['params_dict']['gamma']
    dct['SIR'] = SIR
    dct['fn'] = filenm
    dct['ages'] = {}
    dct['preventions'] = d['preventions']
    dct['prevention_reductions'] = d['prevention_reductions']
    dct['vaccination_dict'] = d['vaccination_dict']
    print("vaccination_dict: ", dct["vaccination_dict"])

    for k,v in ages_SIR.items():
        ages = dct['ages']
        ages[k] = v
    return dct

dicts = []
files = glob.glob("red*")

for filenm in files:
    print("--------------------------------")
    print("filenm: ", filenm)
    dicts.append( readFile(filenm) )

df = pd.DataFrame.from_dict(dicts)
df.to_pickle("metadata.gz")
#df.to_csv("metadata.csv")

print("-------------------")
#readFile("red_mask=0.50,red_dist=0.50,sm=0.50,sd=0.50,wm=0.50,wd=0.50,v=0.99, gamma=0.2, tau=0.2, 2020-10-19,12-37-07")
