# plot a range of runs in different subplots. 
# plot I,L,R curves as a function of latent period.
# Make sure the plot extents are constant. 
# time: [0-80]
# fraction (y): 0-1

import numpy as np
import glob
import os, sys, shutil
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# User libraries
import stats 
import read_data as rd

#----------------------------------------------------------------
run_index = 7    # <<<< Set to create a new run
#----------------------------------------------------------------

# Run this file on a multiple simulation outputs without pandas

L  = 1
IS = 4
R  = 8
PotL  = 10

# base_folder: "r_seir/" for example
def compute_stats(dirs, base_folder):
    for source_folder in list(dirs):
        dest_folder = base_folder + "/" + source_folder.path
        filenm = dest_folder + "/" + "transition_stats.csv"
        try:
            df, IS_L, IS_R, L_IS, IS_PotL = stats.getDataframe(filenm)
            nb_rows = df.shape[0]
        except:
            continue

        print("-----------------------------------------------")
        print("         %s (%d rows)" % (dest_folder, nb_rows))
        stats.processTransmissionTimes(L_IS, "L_IS", plot_data=False)
        stats.processTransmissionTimes(IS_R, "IS_R", plot_data=False)
        stats.processTransmissionTimes(IS_L, "IS_L", plot_data=False)
        stats.individualReproductionNumber(IS_L)

def plot_individual_R(ax, folders, global_dict):
    # Daily average individual reproduction number
    ravg_l = []
    if 1:
        print("folders= ", folders)
        filenm = glob.glob(folders + "/transition_stats.csv")[0]
        print("filenm= %s\n" % filenm)
        df, IS_L, IS_R, L_IS, IS_PotL = stats.getDataframe(filenm)
        _,_,_, df_Ravg    = stats.processTransmissionTimesInTime(IS_L, "IS_L", plot_data=False)
        _,_,_, df_RavgPot = stats.processTransmissionTimesInTime(IS_PotL, "IS_PotL", plot_data=False)
        ravg_l.append({'t': df_Ravg['from_day'], 'R': df_Ravg['avgR'], \
                'tpot': df_RavgPot['from_day'], 'Rpot': df_RavgPot['avgR']})
    
    cols = ['r', 'g', 'b']
    for i, r in enumerate(ravg_l):
        ax.plot(r['t'], r['R'], '-', lw=1)
        ax.plot(r['tpot'], r['Rpot'], '.', lw=1, markersize=2)
    for z in zip(r['R'], r['Rpot']):
        print(z)
    for z in zip(r['t'], r['tpot']):
        print(z)
    ax.set_xlabel('Days')
    ax.set_ylabel('Individual R')
    ax.set_title('Individual R for Configuration model')

def plot_generation_times(df, ax1, ax2, ax3, folders, global_dict):
    isl_avg = []
    isr_avg = []
    ispot_avg = []
    if 1:
        filenm = glob.glob(folders + "/transition_stats.csv")[0]
        df, IS_L, IS_R, L_IS, IS_PotL = stats.getDataframe(filenm)
        # get average time intervals by day
        isl = IS_L.copy()
        isl['time_interval'] = isl['to_time'] - isl['from_time']
        isl['from_day'] = [int(i) for i in isl['from_time']]
        mti = mean_time_intervals = isl.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        isl_avg.append({'t': mti['from_day'], 'time_interval' : mti['time_interval']})

        isr = IS_R.copy()
        isr['time_interval'] = isr['to_time'] - isr['from_time']
        isr['from_day'] = [int(i) for i in isr['from_time']]
        mti = mean_time_intervals = isr.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        isr_avg.append({'t': mti['from_day'], 'time_interval' : mti['time_interval']})

        df_pot = IS_L.copy()
        df_pot = df_pot.append(IS_PotL)
        df_pot['time_interval'] = df_pot['to_time'] - df_pot['from_time']
        df_pot['from_day'] = [int(i) for i in df_pot['from_time']]
        mti = mean_time_intervals = df_pot.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        ispot_avg.append({'t': mti['from_day'], 'time_interval' : mti['time_interval']})


    cols = ['r', 'g', 'b']
    for i,r in enumerate(isl_avg):
        ax1.plot(r['t'], r['time_interval'], color=cols[i], lw=1)

    for i,r in enumerate(isr_avg):
        ax2.plot(r['t'], r['time_interval'], color=cols[i], lw=1)

    for i,r in enumerate(ispot_avg):
        ax3.plot(r['t'], r['time_interval'], color=cols[i], lw=1)

    ax1.set_title("Time interval IS-L")
    ax2.set_title("Time interval IS-R")
    ax3.set_title("Time interval pot_real IS-L")

def plot_infections_by_degree(delta_time, ax, folder, global_dict):
    if 1:
        col = global_dict['color']
        run = global_dict['run']
        vac1_rate = global_dict['vac1_rate']
        filenm = glob.glob(folder + "/data_baseline_p0.txt")[0]
        print("filenm= ", filenm)
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run0 = 0
        latent, infected, recov = rd.get_SEIR(by, run0, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(infected)))
        N = 260542  # hardcoded. BAD
        lw = 0.5
        #ax.plot(times, infected/N, color=col, lw=lw, label=f"{vac1_rate}")
        #ax.plot(times, latent/N, color=col, lw=lw) # label="Latent")
        ax.plot(times, (infected+latent)/N, color=col, lw=lw, label=f"{vac1_rate}") #, label="IS+L")
        ax.plot(times, recov/N, color=col, lw=lw) #, label="R")

    # plot vaccinated
    # time,S,L,IS,R,vacc1,vacc2
    filenm = folder + "/counts.csv"
    df = pd.read_csv(filenm)
    IS = df["IS"].values
    V1 = df["vacc1"].values
    V2 = df["vacc2"].values
    ti = df["time"].values
    ax.plot(ti, V1/N, color=col, lw=1, ls='--')
    ax.plot(ti, V2/N, color=col, lw=1, ls='--')
    ax.set_xlim(0,80)
    ax.set_ylim(0,1)
    ax.grid(True)
    ax.legend(title="Daily vacc rate\n(1st dose)", loc='upper right')
    eps_S = global_dict["epsilonS-1"]
    ax.set_title(f"I, L, R, curves (latent period:{eps_S} days\n(frac of tot pop size)")
    ax.set_xlabel("Time")

def plot_latent_by_degree(delta_time, ax, folders, global_dict):
    cols = ['r', 'g', 'b']
    if 1:
        filenm = glob.glob(folders + "/data_baseline_p0.txt")[0]
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run = 0
        print("==> run= ", run)
        latent, infected, recov = rd.get_SEIR(by, run, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(latent)))
        ax.plot(times, latent)
    ax.set_title("Nb Latent")


#----------------------------------------------------------------------
if __name__ == "__main__":

    # This script presumes a loop over a degrees list (hardcoded in the various methods)

    base_run_folder = "run%03d" % run_index    # <<<< MUST BE SET
    base_folder = "data_ge/%s/results_run%03d" 
    runs = range(4)

    # one color per case
    cols = ['r', 'g', 'b', 'c', 'm', 'orange']

    # Store the plots in variables. Rearrange later if possible.

    folders = []
    dirs = list(os.scandir("graphs"))
    
    dt = 0.1  # time step in units of days
    r, c = 2,2
    fig, axes = plt.subplots(r, c, figsize=(10,10))
    axes = np.asarray(axes).reshape(-1)
    print("=========> axes= ", axes)
    print("folders= ", folders)

    #for ic,case in enumerate([4,7,6,5]):
    for ic,case in enumerate([8,9,10,11]):
      base_run_folder = "run%03d" % case    # <<<< MUST BE SET
      base_folder = "data_ge/%s/results_run%03d" 
      ax = axes[ic]
      base_run = case

      for run in runs:
        print(run)
        folder = base_folder % (base_run_folder, run)
        with open(folder + "/global_dict.pkl", "rb") as f:
            global_dict = pickle.load(f)
        global_dict['colors'] = cols
        global_dict['color'] = cols[run]
        global_dict['run'] = run
        plot_infections_by_degree(dt, ax, folder, global_dict)

    plt.tight_layout()
    plt.savefig('plot_individual_R.pdf')

