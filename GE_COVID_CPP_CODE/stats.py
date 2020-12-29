# Author: Gordon Erlebacher
# Date: 2020-12-19
# Compute timing distributions based on C++ code results for SEIR

# Use Leon County Graph as well as synthetic graphs

import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
#import read_data as rd



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
# Non-susceptible nodes that cannot get infected
#define PotL 10
"""

L  = 1
IS = 4
R  = 8
PotL  = 10


" columns: from_id, to_id, from_state, to_state, from_time, to_time"

# computer overall generation distribution. 
# Identify IS -> L and L -> IS transitions

def getDataframe(filenm):
    df = pd.read_csv(filenm)
    by = df.groupby(["from_state", "to_state"])
    IS_L = by.get_group((IS, L))  # two different ids (wrong results)
    IS_R = by.get_group((IS, R))  # same ids
    L_IS = by.get_group((L, IS))  # same ids
    IS_PotL = by.get_group((IS, PotL))
    print("len(IS_L): ", IS_L.shape)
    print("len(IS_PotL): ", IS_PotL.shape)
    return df, IS_L, IS_R, L_IS, IS_PotL

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
    plt.savefig("plot_degree_distribution.pdf")

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
    varR0 = np.var(list(Rd.values()))
    print("average R0: %f, var R0: %f" % (R0, varR0))

    hist = defaultdict(int)
    for v in Rd.values():
        hist[v] += 1
    hist = np.asarray(sorted(hist.items()))

    #print("Rd Distribution: \n", hist)
    #print(hist)

    #plt.hist(Rd.values())
    #plt.xlim(0,30)
    #plt.show()
    #plt.savefig("plot_indiv_R.pdf")

#---------------------------------------------
def processTransmissionTimes(df, label, plot_data=False):

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

    return times

#----------------------------------------------------------------------
def processTransmissionTimesInTime(df, label, plot_data=False):
    # Compute mean time between IS and L in different time intervals T=[t,t+1day]
    # Collect the nodes that become latent within T, and use these to compute the 
    # mean generation time. The input dataframe (df) must only contain transitions
    # from IS to L

    #print("Inside process TransmissionTimes")
    from_time = df['from_time'].values
    to_time   = df['to_time'].values

    # Add a column to identify the day corresponding to from_time
    df1 = df.copy()
    df1['from_day'] = [int(i) for i in from_time]
    df1['time_interval'] = to_time - from_time
    days = df1.groupby('from_day').agg({'time_interval': ['mean','count']})

    # calculate prevalance: number of infected each day
    # Of interest: 'time_interval', which is averaged over an entire day
    df2 = df1.groupby(['from_id', 'from_state']).mean()

    # calculate individual reproduction number by day
    df_Rindiv = df1.groupby(['from_id']).agg({'from_day':'mean', 'from_state':'mean', 'to_state':'count'}).rename(columns={'to_state':'nb_of_to_state'}).reset_index()

    # calculate averate daily reproduction number
    df_Ravg   = df_Rindiv.groupby(['from_day']).agg({'nb_of_to_state': 'mean'}).rename(columns={'nb_of_to_state':'avgR'}).reset_index()


    # Compute incidence: number of new cases each day
    df3 = df2.groupby('from_day').count().reset_index() # return days to colset

    # Compute individual reproduction number in time
    # Each day, identify newly infected. How many people do these infected infect
    #  (count the number of people that go from IS to L states

    # df3['to_id'] is now the incidence of infected
    # To get the cumulative sum cannot be done from here. 

    if plot_data:
        plt.plot(df_Ravg['from_day'], df_Ravg['avgR'],'.-')
        plt.xlabel('days')
        plt.ylabel('R_d(t)')
        plt.title("Average Individual Reproduction Number")
        plt.savefig("plot_avg_indiv_reprod_number.pdf")


    if plot_data:
        fig, ax = plt.subplots()
        ax.plot(days['time_interval', 'mean'], color='r')
        ax.set_xlabel("Days")
        ax.set_ylabel("Mean Interval L-IS (days)", color='r', fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(days['time_interval', 'count'], color='b')
        ax2.set_ylabel("Nb of intervals used", color='b', fontsize=14)
        # all columns have same value
        ax2.plot(df3['from_day'], df3['to_id'], color='cyan', label='incidence') 
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_hist_mean_interv.pdf")

    #print("df_Ravg= \n", df_Ravg)
    return df1, df2, df3, df_Ravg

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

    #for k,v in times_d.items():
        #print(k,v)

#----------------------------------------------------------------------
if __name__ == "__main__":
    filenm = glob.glob('r_seir/graphs/*_10/transition_stats.csv')[0]
    print(filenm)
    df, IS_L, IS_R, L_IS, IS_PotL = getDataframe(filenm)

    processTransmissionTimesInTime(IS_L, "IS_L", plot_data=False)
    processTransmissionTimesInTime(IS_PotL, "IS_PotL", plot_data=False)
    processTransmissionTimes(L_IS, "L_IS", plot_data=False)
    processTransmissionTimes_2nd_method(df)

    degreeDistribution()
    individualReproductionNumber(IS_L)

    print("Distribution: from Infected to Latent")
    r, c = 2,2
    plot_data = True
    plt.subplots(r,c, figsize=(10,8))
    plt.subplot(r,c,1)
    processTransmissionTimes(IS_L, "IS_L", plot_data=plot_data)
    plt.subplot(r,c,2)
    processTransmissionTimes(IS_R, "IS_R", plot_data=plot_data)
    plt.subplot(r,c,3)
    processTransmissionTimes(L_IS, "L_IS", plot_data=plot_data)
    plt.subplot(r,c,4)
    processTransmissionTimes(IS_PotL,  "IS_PotL", plot_data=plot_data)

    # Add IS_PotL and IS_L together
    print(IS_L.shape, IS_PotL.shape)
    IS_Tot = IS_L.append(IS_PotL)
    processTransmissionTimes(IS_PotL,  "IS_PotL", plot_data=plot_data)
    processTransmissionTimes(IS_Tot,  "IS_Tot", plot_data=plot_data)

    print("----------------------------------------")
    print("Shapes: IS_L, IS_PotL, IS_Tot")
    print(IS_L.shape, IS_PotL.shape, IS_Tot.shape)
    plt.savefig("plot_transmission_times_hist.pdf")
#----------------------------------------------------
