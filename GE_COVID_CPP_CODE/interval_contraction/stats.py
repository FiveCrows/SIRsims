# Author: Gordon Erlebacher
# Date: 2020-12-19
# Compute timing distributions based on C++ code results for SEIR

# Use Leon County Graph as well as synthetic graphs

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt



"""
//States
#define S 0
#define L 1
#define IA 2
#define PS 3
#define IS 4
#define HOME 5
#define H 6
#define ICU 7
#define R 8
"""

L  = 1
IS = 4
R  = 8


" columns: from_id, to_id, from_state, to_state, from_time, to_time"

# computer overall generation distribution. 
# Identify IS -> L and L -> IS transitions

def getDataframe(filenm):
    df = pd.read_csv(filenm)
    by = df.groupby(["from_state", "to_state"])
    IS_L = by.get_group((IS, L))  # two different ids (wrong results)
    IS_R = by.get_group((IS, R))  # same ids
    L_IS = by.get_group((L, IS))  # same ids
    return df, IS_L, IS_R, L_IS

# Compute generation distributions
# IS -> L (wrong results)
# IS -> R
#  L -> IS


#-----------------------------------------------
def degreeDistribution():
    # compute the degree distribution of the input graph
    filenm = "Data_BA/network1.txt"
    filenm = "Data_random/network.txt"
    txt = np.loadtxt(filenm, usecols=[0,1], dtype='int')

    N = txt.shape[0]
    deg = defaultdict(int)
    for i in range(N):
        i,j = txt[i,:]
        deg[i] += 1
        deg[j] += 1

    # deg = degree of each node
    deg_hist = defaultdict(int)
    for k,v in deg.items():
        deg_hist[v] += 1

    deg_hist = np.asarray(sorted(deg_hist.items()))
    print("Degree Distribution: \n", deg_hist)
    plt.bar(deg_hist[:,0], deg_hist[:,1])
    plt.xlim(0,30)
    plt.show()

#---------------------------------------------
def individualReproductionNumber(df):
    from_id = df['from_id'].values
    to_id   = df['to_id'].values

    # individual reproduction number
    Rd = defaultdict(int)

    for fid in from_id:
        Rd[fid] += 1

    # Average R0 across time
    R0 = np.mean(list(Rd.values()))
    print("average R0: ", R0)

    hist = defaultdict(int)
    for v in Rd.values():
        hist[v] += 1
    hist = np.asarray(sorted(hist.items()))
    print("Rd Distribution: \n", hist)
    print(hist)

    plt.hist(Rd.values())
    plt.xlim(0,30)

    plt.show()

#---------------------------------------------
def processTransmissionTimes(df, label, plot_data=False):

    keep = df['from_time'] < 10
    from_time = df['from_time'].values
    to_time   = df['to_time'].values
    #from_id   = df['from_id'].values
    #to_id     = df['to_id'].values

    lg = from_time.shape[0]
    if lg < 3: return

    nb = -1
    from_time = from_time[0:nb]
    to_time   = to_time[0:nb]

    # time between two events as determined by df selected in arg list
    times = to_time - from_time

    print("(%s,lg=%d), Sample mean/var= %f, %f" % (label, lg, np.mean(times), np.var(times)))

    if plot_data:
        plt.title(label)
        plot = plt.hist(times, bins=200)
        plt.xlim(0,20)
        return plot
    else:
        return label, np.mean(times), np.var(times)


#---------------------------------------------
def processTransmissionTimes_2nd_method(df):
    from_time = df['from_time'].values
    to_time   = df['to_time'].values
    from_id   = df['from_id'].values
    to_id     = df['to_id'].values
    from_s    = df['from_state'].values
    to_s      = df['to_state'].values

    times_d = defaultdict(list)
    times = to_time - from_time

    for i in range(df.shape[0]):
        #times_d[to_id[i]].append(times[i])
        times_d[to_id[i]].append((times[i], (from_time[i], to_time[i]), to_s[i]))

    #3 - 4 ==> 7.     IS -> L
    #4 - 4 ==> 3.     L -> IS
    #4 - 4 ==> 2.  
    #times[4] = [7, 3, 2]

    for k,v in times_d.items():
        print(k,v)

#----------------------------------------------------------------------
if __name__ == "__main__":
    filenm = 'transition_stats.csv'
    df, IS_L, IS_R, L_IS = getDataframe(filenm)

    processTransmissionTimes(L_IS, "L_IS", plot_data=False)
    processTransmissionTimes_2nd_method(df)

    degreeDistribution()
    individualReproductionNumber(df)

    print("Distribution: from Infected to Latent")
    plt.subplots(2,2)
    plt.subplot(2,2,1)
    processTransmissionTimes(IS_L, "IS_L")
    plt.subplot(2,2,2)
    processTransmissionTimes(IS_R, "IS_R")
    plt.subplot(2,2,3)
    processTransmissionTimes(L_IS, "L_IS")
    #plt.subplot(2,2,4)
    #processTransmissionTimes(S_L,  "S_L")
    plt.show()
#----------------------------------------------------
