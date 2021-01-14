import numpy as np
import glob
import os, sys, shutil
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
import glob

# User libraries
import stats 
import read_data as rd
from timings import *


#----------------------------------------------------------------
run_index = [14]    # <<<< Process one or multiple runs
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

#-----------------------------------------------------------
@timeit
def plot_individual_R(ax, axc, folder, global_dict):
    # Daily average individual reproduction number
    ravg_l = []
    if 1:
        IS_L = pd.read_csv(f"{folder}/IS_L.csv")
        R_avg = pd.read_csv(f"{folder}/R_avg.csv")
        ravg_l.append({'t': R_avg['from_day'], 'R': R_avg['R_avg'], 'count': R_avg['R_count']})
    
    vacc1_rate = global_dict['vacc1_rate']
    max_nb_avail_doses = global_dict["max_nb_avail_doses"]
    epsilonSinv = global_dict["epsilonSinv"]

    col = global_dict['color']
    for i, r in enumerate(ravg_l):
        ax.plot(r['t'], r['R'], '-', lw=1, color=col,
                 label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")
        axc.plot(r['t'], r['count'], '-', lw=1, color=col,
                 label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")

    ax.axhline(y=1.0)
    ax.set_xlabel('Days')
    ax.set_ylabel('Avg Indiv. R(t)')
    ax.set_title('Individual R')
    leg_text = "Daily vacc rate (dose 1)\nmax nb avail doses\nlatent time"
    ax.legend(fontsize=8, title=leg_text, loc='upper right')

    axc.set_xlabel('Days')
    axc.set_ylabel('Avg Indiv. R count')
    axc.set_title('Nb of samples of indiv R(t)')
    axc.legend(fontsize=8, title=leg_text, loc='upper right')



#----------------------------------------------------------
@timeit
def plot_generation_times_2(ax1, ax2, ax3, folder, global_dict): #, IS_L, IS_R, L_IS):
    isl_avg = []
    ili_avg = []
    isr_avg = []
    ispot_avg = []
    if 1:
        #filenm = glob.glob(f"{folder}/transition_stats.csv")[0]
        IS_L = pd.read_csv(f"{folder}/IS_L.csv")
        IS_R = pd.read_csv(f"{folder}/IS_R.csv")
        L_IS = pd.read_csv(f"{folder}/L_IS.csv")
        isl_avg.append({'t': IS_L['from_day'], 'time_interval' : IS_L['time_interval_mean']})
        isr_avg.append({'t': IS_R['from_day'], 'time_interval' : IS_R['time_interval_mean']})
        ili_avg.append({'t': L_IS['from_day'], 'time_interval' : L_IS['time_interval_mean']})

    col = global_dict['color']
    lw = 0.5
    vacc1_rate = global_dict['vacc1_rate']
    max_nb_avail_doses = global_dict["max_nb_avail_doses"]
    epsilonSinv = global_dict["epsilonSinv"]

    for i,r in enumerate(isl_avg):
        ax1.plot(r['t'], r['time_interval'], color=col, lw=lw, 
                label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")

    for i,r in enumerate(isr_avg):
        ax2.plot(r['t'], r['time_interval'], color=col, lw=lw) 

    for i,r in enumerate(ili_avg):
        ax3.plot(r['t'], r['time_interval'], color=col, lw=lw)

    ax1.set_title("Time interval IS-L")
    ax2.set_title("Time interval IS-R")
    ax3.set_title("Time interval IL_I")
    #ax3.set_title("Time interval pot_real IS-L")
    lg = ax1.legend(fontsize=8, title="Daily vacc rate (dose 1)\n \
             max nb avail doses\n \
             latent time", loc='upper right')

#----------------------------------------------------------
def plot_infections_by_degree(delta_time, ax, axv, folder, global_dict):
    if 1:
        col = global_dict['color']
        N = global_dict['N']  
        run = global_dict['run']
        vacc1_rate = global_dict['vacc1_rate']
        filenm = glob.glob(folder + "/data_baseline_p0.txt")[0]
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run0 = 0
        latent, infected, recov = rd.get_SEIR(by, run0, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(infected)))
        #N = 260542  # hardcoded. BAD
        lw = 0.5
        ax.plot(times, infected/N, color=col, lw=lw) #, label=f"{vacc1_rate}") 
        ax.plot(times, recov/N, color=col, lw=lw) #, label="R")
        eps_S = global_dict["epsilonS-1"]
        ax.set_title(f"I curves (latent period:{eps_S} days\n(frac of tot pop size)")

    # plot vaccinated
    # time,S,L,IS,R,vacc1,vacc2
    filenm = folder + "/counts.csv"
    df = pd.read_csv(filenm)
    IS = df["IS"].values
    V1 = df["vacc1"].values
    V2 = df["vacc2"].values
    ti = df["time"].values
    axv.plot(ti, V1/N, color=col, lw=1, ls='--')
    axv.plot(ti, V2/N, color=col, lw=1, ls='--')
    axv.legend(title="Daily vacc rate\n(1st dose)", loc='upper right')
    axv.set_title(f"Number of people vaccinated\n1st and 2nd dose")
    axv.set_xlabel("Time")

def plot_latent_by_degree(delta_time, ax, folders, global_dict):
    if 1:
        col = global_dict['color']
        N = global_dict['N']  
        run = global_dict['run']
        run0 = 0
        print("run= ", run)
        print("-- folders= ", folders)
        filenm = glob.glob(folders + "/data_baseline_p0.txt")[0]
        df = rd.setupDataframe(filenm)
        print(df.columns)
        print("df= ", df['run'])
        by = df.groupby("run")
        # run0 is last column of the data file (not used)
        latent, infected, recov = rd.get_SEIR(by, run0, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(latent)))
        lw = 0.5
        ax.plot(times, latent/N, color=col, lw=lw)
    ax.set_title("Nb Latent")


#----------------------------------------------------------------------
if __name__ == "__main__":

    # This script presumes a loop over a degrees list (hardcoded in the various methods)

    #base_run_folder = "run%05d" % run_index[0]    # <<<< MUST BE SET

    #base_folder = "data_ge/%s/results_run%04d" 
    #runs = range(4)
    runs = range(8)

    # one color per case
    cols = ['r', 'g', 'b', 'c', 'm', 'orange', 'black']

    # Store the plots in variables. Rearrange later if possible.

    dt = 0.1  # time step in units of days
    r, c = 4,2
    fig, axes = plt.subplots(r, c, figsize=(12,10))
    #axes = np.asarray(axes).reshape(-1)
    print("=========> axes= ", axes)

    for ic,case in enumerate(run_index):
      folders = glob.glob("data_ge/run*%05d/result*/" % case)
      nb_folders = len(folders)
      print(folders)

      for run in range(nb_folders):
        print("--> run: ", run)
        folder = "data_ge/" + "run%05d" % case + "/" + "results_run%04d" % run 
        print("folder1: ", folder)
        with open(folder + "/global_dict.pkl", "rb") as f:
            global_dict = pickle.load(f)
        global_dict['colors'] = cols
        global_dict['color'] = cols[run % len(cols)]
        global_dict['run'] = run
        # Should automate next three lines
        print("vacc1_rate: ", global_dict["vacc1_rate"])
        print("max_nb_avail_doses: ", global_dict["max_nb_avail_doses"])
        print("epsilonSinv: ", global_dict["epsilonSinv"])
        # Individual R and counts of individuals infected (symptomatic) each day (incidence)
        plot_individual_R(axes[0,0], axes[1,0], folder, global_dict)
        plot_infections_by_degree(dt, axes[0,1], axes[1,1], folder, global_dict)
        plot_latent_by_degree(dt, axes[2,0], folder, global_dict)
        plot_generation_times_2(axes[2,1], axes[3,0], axes[3,1], folder, global_dict)

    plt.tight_layout()
    plt.savefig('plot_multi_generation_times.pdf')

