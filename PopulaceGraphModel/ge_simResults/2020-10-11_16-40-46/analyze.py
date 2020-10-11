import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import scanf
import pandas as pd
from scipy.interpolate import interp1d

files = glob.glob("red*")
filenm = "red_mask=0.00,red_dist=0.10,sm=1,sd=1,wm=0,wd=1, gamma=0.2, tau=0.2, 2020-10-11,02.39pm"
red_mask, red_dist, sm, sd, wm, wd, gamma, tau = scanf.scanf("red_mask=%f,red_dist=%f,sm=%d,sd=%d,wm=%d,wd=%d, gamma=%f, tau=%f", filenm)
print(red_mask, red_dist, sm, sd, wm, wd, gamma, tau)

def interpolate_SIR(SIR):
    S = SIR['S']
    I = SIR['I']
    R = SIR['R']
    t = SIR['t']
    # interpolate on daily intervals. 
    new_t = np.linspace(0., int(t[-1]), int(t[-1])+1)
    func = interp1d(t, S) 
    Snew = func(new_t)
    func = interp1d(t, I)
    Inew = func(new_t)
    func = interp1d(t, R)
    Rnew = func(new_t)
    #print("t= ", new_t)
    #print("S= ", Snew)
    #print("I= ", Inew)
    #print("R= ", Rnew)
    SIR['t'] = new_t
    SIR['S'] = Snew
    SIR['I'] = Inew
    SIR['R'] = Rnew
    return SIR

def readFile(filenm):
    fd = open(filenm, "rb")
    d = pickle.load(fd)
    ages_SIR = d['ages_SIR']
    SIR = d['sim_results']
    #SIR = interpolate_SIR(SIR)
    #print("len: SIR.S= ", len(SIR['S']))
    #print("ages_SIR.keys()= ", list(ages_SIR.keys()))

    red_mask, red_dist, sm, sd, wm, wd = scanf.scanf("red_mask=%f,red_dist=%f,sm=%d,sd=%d,wm=%d,wd=%d", d['title'])
    dct = {'sm':sm, 'sd':sd, 'wm':wm, 'wd':wd, 'red_mask': red_mask, 'red_dist': red_dist}
    dct['tau'] = d['params']['tau']
    dct['gamma'] = d['params']['gamma']
    dct['SIR'] = SIR
    dct['ages'] = {}
    for k,v in ages_SIR.items():
        ages = dct['ages']
        ages[k] = v
    return dct

# top 20 files
#for filenm in files[0:20]:
dicts = []
for filenm in files:
    dicts.append( readFile(filenm) )

df = pd.DataFrame.from_dict(dicts)
df.to_csv("metadata.csv")
df.to_pickle("metadata.gz")

