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
    SIR = d['sim_results']
    SIR = interpolate_SIR(SIR)

    d['params'] = d['params'][0]  # the zero is not really required (I fixed a bug)
    red_mask, red_dist, sm, sd, wm, wd = scanf.scanf("red_mask=%f,red_dist=%f,sm=%d,sd=%d,wm=%d,wd=%d", d['title'])
    return red_mask, red_dist, sm, sd, wm, wd, d['params']['gamma'], d['params']['tau='] # FIX 'tau'. FIXED. 

# I need a more general method that does not depend on creating this list
sml = []
sdl = []
wml = []
wdl = []
red_maskl = []
red_distl = []
filenml = []
gammal = []
taul = []

for filenm in files[0:20]:
    print(filenm)
    d = readFile(filenm)
    filenml.append(filenm)
    red_mask, red_dist, sm, sd, wm, wd, gamma, tau = readFile(filenm)
    sml.append(sm)
    sdl.append(sd)
    wml.append(wm)
    wdl.append(wd)
    red_maskl.append(red_mask)
    red_distl.append(red_dist)
    gammal.append(gamma)
    taul.append(tau)

columns = ['sm','sd','wm','wd','red_mask', 'red_dist','gamma', 'tau', 'fn']
arrays = [sml,sdl,wml,wdl,red_maskl,red_distl,gammal,taul, filenml]
df = pd.DataFrame(arrays).transpose()
df.columns = columns
df.to_pickle("metadata.gz")
ddf = pd.read_pickle("metadata.gz")

