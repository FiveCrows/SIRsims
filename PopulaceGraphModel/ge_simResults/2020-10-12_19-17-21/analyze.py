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

    red_mask, red_dist, sm, sd, wm, wd = scanf.scanf("red_mask=%f,red_dist=%f,sm=%d,sd=%d,wm=%d,wd=%d", d['title'])
    dct = {'sm':sm, 'sd':sd, 'wm':wm, 'wd':wd, 'red_mask': red_mask, 'red_dist': red_dist}
    dct['tau'] = d['params']['tau']
    dct['gamma'] = d['params']['gamma']
    dct['SIR'] = SIR
    dct['fn'] = filenm
    dct['ages'] = {}
    for k,v in ages_SIR.items():
        ages = dct['ages']
        ages[k] = v
    return dct

# top 20 files
#for filenm in files[0:20]:
dicts = []
files = glob.glob("red*")

for filenm in files:
    dicts.append( readFile(filenm) )

df = pd.DataFrame.from_dict(dicts)
df.to_pickle("metadata.gz")

