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

xlim_max = 40


#----------------------------------------------------------------
# specify project number via command line argument

nb_arguments = len(sys.argv) -1
if nb_arguments == 1:
    print("Project: ", sys.argv[1])
    run_index = [int(sys.argv[1])]
else:
    print("missing argument, EXIT")
    run_index = -1 # force error
    quit()
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
    lw = 0.5


    col = global_dict['color']
    alpha=0.7

    for i, r in enumerate(ravg_l):
        h, = ax.plot(r['t'], r['R'], '-', lw=lw, color=col, alpha=alpha,
                 label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")
        hc, = axc.plot(r['t'], r['count'], '-', lw=lw, color=col, alpha=alpha,
                 label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")

    print("plot_individualR, col= %s, repeat: %d, vacc1_rate: %f, max_nb_avail_doses: %d, run: %d, repeat_run: %d" % (col, global_dict["repeat_run"], global_dict["vacc1_rate"], global_dict["max_nb_avail_doses"], global_dict["run"], global_dict["repeat_run"]))

    ax.axhline(y=1.0)
    ax.set_xlabel('Days')
    ax.set_ylabel('Avg Indiv. R(t)')
    ax.set_title('Individual R')
    ax.set_xlim(0,xlim_max)
    ax.set_ylim(0,8)
    ax.grid(True)
    #leg_text = "Daily vacc rate (dose 1)\nmax nb avail doses\nlatent time"
    #ax.legend(fontsize=6, title=leg_text, loc='upper right')

    axc.set_xlabel('Days')
    axc.set_ylabel('Avg Indiv. R count')
    axc.set_title('Nb of samples of indiv R(t)')
    axc.set_xlim(0,xlim_max)
    axc.grid(True)
    return h, hc  # handles

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
    alpha = 0.7
    vacc1_rate = global_dict['vacc1_rate']
    max_nb_avail_doses = global_dict["max_nb_avail_doses"]
    epsilonSinv = global_dict["epsilonSinv"]

    for i,r in enumerate(isl_avg):
        ax1.plot(r['t'], r['time_interval'], color=col, lw=lw, alpha=alpha,
                label=f"{vacc1_rate},{max_nb_avail_doses},{epsilonSinv}")

    for i,r in enumerate(isr_avg):
        ax2.plot(r['t'], r['time_interval'], color=col, lw=lw, alpha=alpha)

    for i,r in enumerate(ili_avg):
        ax3.plot(r['t'], r['time_interval'], color=col, lw=lw, alpha=alpha)

    ax1.set_title("Time interval IS-L")
    ax2.set_title("Time interval IS-R")
    ax3.set_title("Time interval IL_I")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.set_xlim(0,xlim_max)
    ax2.set_xlim(0,xlim_max)
    ax3.set_xlim(0,xlim_max)
    ax1.set_ylim(0,7)
    ax2.set_ylim(0,7)
    ax3.set_ylim(0,10)
    #ax3.set_title("Time interval pot_real IS-L")

    """
    lg = ax1.legend(fontsize=6, title="Daily vacc rate (dose 1)\n \
             max nb avail doses\n \
             latent time", loc='upper right')
    """

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
        alpha = 0.7
        ax.plot(times, infected/N, color=col, lw=lw) #, label=f"{vacc1_rate}") 
        ax.plot(times, recov/N, color=col, lw=lw, alpha=alpha) #, label="R")
        ax.grid(True)
        ax.set_xlim(0,xlim_max)
        eps_S = global_dict["epsilonS-1"]
        ax.set_title(f"Frac Infected (latent period:{eps_S} days\n(frac of tot pop size)")

    # plot vaccinated
    # time,S,L,IS,R,vacc1,vacc2
    filenm = folder + "/counts.csv"
    df = pd.read_csv(filenm)
    IS = df["IS"].values
    V1 = df["vacc1"].values
    V2 = df["vacc2"].values
    ti = df["time"].values
    axv.plot(ti, V1, color=col, lw=1, ls='--', alpha=alpha)
    axv.plot(ti, V2, color=col, lw=1, ls='--', alpha=alpha)
    axv.legend(title="Daily vacc rate\n(1st dose)", loc='upper right')
    axv.grid(True)
    axv.set_title(f"Number of people vaccinated\n1st and 2nd dose")
    axv.set_xlabel("Time")
    axv.set_xlim(0,xlim_max)

def plot_latent_by_degree(delta_time, ax, folders, global_dict):
    if 1:
        alpha = 0.7
        lw = 0.5
        col = global_dict['color']
        N = global_dict['N']  
        run = global_dict['run']
        run0 = 0
        #print("run= ", run)
        #print("-- folders= ", folders)
        filenm = glob.glob(folders + "/data_baseline_p0.txt")[0]
        df = rd.setupDataframe(filenm)
        #print(df.columns)
        #print("df= ", df['run'])
        by = df.groupby("run")
        # run0 is last column of the data file (not used)
        latent, infected, recov = rd.get_SEIR(by, run0, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(latent)))
        ax.plot(times, latent/N, color=col, lw=lw, alpha=alpha)
    ax.set_xlim(0,xlim_max)
    ax.set_title("Fraction Latent")
    ax.grid(True)


#----------------------------------------------------------------------
def plot_infection_curve(ax1, ax2, folder, global_dict)
    pass
#----------------------------------------------------------------------
if __name__ == "__main__":

    # This script presumes a loop over a degrees list (hardcoded in the various methods)

    # one color per case
    cols = ['r', 'g', 'b', 'c', 'm', 'orange', 'black']

    # Store the plots in variables. Rearrange later if possible.

    dt = 0.1  # time step in units of days
    r, c = 5,2
    fig, axes = plt.subplots(r, c, figsize=(12,10))
    #axes = np.asarray(axes).reshape(-1)

    handles_R = {}
    hd = {}; hdc = {}

    # Run all or a subset of files in a given project

    for ic,case in enumerate(run_index):
      folders = glob.glob("data_ge/project*%05d/result*/" % case)
      nb_folders = len(folders)

      for run in range(nb_folders):
        #print("--> run: ", run)
        folder = "data_ge/" + "project%05d" % case + "/" + "results_run%04d" % run 
        #print("folder1: ", folder)
        with open(folder + "/global_dict.pkl", "rb") as f:
            global_dict = pickle.load(f)
        global_dict['colors'] = cols
        try:
          global_dict['color'] = cols[global_dict['top_level_run'] % len(cols)]
        except:
          global_dict['color'] = 'b'
        global_dict['run'] = run
        #print(global_dict)
        #print("top_level_run: ", global_dict["top_level_run"])
        # Should automate next three lines
        #print("vacc1_rate: ", global_dict["vacc1_rate"])
        #print("max_nb_avail_doses: ", global_dict["max_nb_avail_doses"])
        #print("epsilonSinv: ", global_dict["epsilonSinv"])

        # Individual R and counts of individuals infected (symptomatic) each day (incidence)
        h, hc = plot_individual_R(axes[0,0], axes[1,0], folder, global_dict)
        hd[global_dict['top_level_run']] = h
        hdc[global_dict['top_level_run']] = hc
        #print("hd= ", hd)

        plot_infections_by_degree(dt, axes[0,1], axes[1,1], folder, global_dict)
        plot_latent_by_degree(dt, axes[2,0], folder, global_dict)
        plot_generation_times_2(axes[2,1], axes[3,0], axes[3,1], folder, global_dict)

        # Plot infectivity curve
        plot_infection_curve(axes[4,0], axes[4,1], folder, global_dict)

    hd  = list(hd.values()); hdc = list(hdc.values())
    fsz = 8
    leg_text = "Daily vacc rate (dose 1)\nmax nb avail doses\nlatent time"
    lg = axes[0,0].legend(handles=hd, title=leg_text, loc='upper right', ncol=1, fontsize=fsz)
    lgc = axes[1,0].legend(handles=hdc, title=leg_text, loc='upper right', ncol=1, fontsize=fsz)
    #print(help(lg.set_title))
    lg.get_title().set_fontsize(str(fsz))
    lgc.get_title().set_fontsize(str(fsz))
    plt.tight_layout()
    plt.savefig(f"./plot_multi_generation_times_project{case}.pdf")

