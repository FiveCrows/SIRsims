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
run_index = [19]    # <<<< Process one or multiple runs
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
    
    R0 = global_dict['R0']
    beta_shape = global_dict["beta_shape"]
    beta_scale = global_dict["beta_scale"]
    label=f"{R0},{beta_shape},{beta_scale}"

    lw = 0.5


    col = global_dict['color']
    ltype = global_dict['ltype']
    alpha=0.7
    marker = global_dict['marker']
    xlim = 40.0

    for i, r in enumerate(ravg_l):
        h, = ax.plot(r['t'], r['R'], '-', lw=lw, ls=ltype, color=col, marker=marker, ms=0.5, alpha=alpha, label=label)
        hc, = axc.plot(r['t'], r['count'], '-', lw=lw, ls=ltype, color=col, marker=marker, ms=0.5, alpha=alpha, label=label)

    ax.axhline(y=1.0)
    ax.set_xlabel('Days')
    ax.set_ylabel('Avg Indiv. R(t)')
    ax.set_title('Individual R')
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,8)
    ax.grid(True)
    #leg_text = "Daily vacc rate (dose 1)\nmax nb avail doses\nlatent time"
    #ax.legend(fontsize=6, title=leg_text, loc='upper right')

    axc.set_xlabel('Days')
    axc.set_ylabel('Avg Indiv. R count')
    axc.set_title('Nb of samples of indiv R(t)')
    axc.set_xlim(0,xlim)
    axc.grid(True)
    return h, hc  # handles

#----------------------------------------------------------
@timeit
def plot_generation_times_2(ax1, ax2, ax3, folder, global_dict): #, IS_L, IS_R, L_IS):
    isl_avg = []
    ili_avg = []
    ill_avg = []
    ispot_avg = []
    if 1:
        #filenm = glob.glob(f"{folder}/transition_stats.csv")[0]
        IS_L = pd.read_csv(f"{folder}/IS_L.csv")
        IL_L = pd.read_csv(f"{folder}/L_L.csv")
        L_IS = pd.read_csv(f"{folder}/L_IS.csv")
        isl_avg.append({'t': IS_L['from_day'], 'time_interval' : IS_L['time_interval_mean']})
        ill_avg.append({'t': IL_L['from_day'], 'time_interval' : IL_L['time_interval_mean']})
        ili_avg.append({'t': L_IS['from_day'], 'time_interval' : L_IS['time_interval_mean']})

    col = global_dict['color']
    marker = global_dict['marker']
    lw = 0.5
    alpha = 0.7
    xlim = 40.0
    R0 = global_dict['R0']
    lstyle = global_dict['ltype']
    beta_shape = global_dict["beta_shape"]
    beta_scale = global_dict["beta_scale"]
    label=f"{R0},{beta_shape},{beta_scale}"

    for i,r in enumerate(isl_avg):
        ax1.plot(r['t'], r['time_interval'], color=col, lw=lw, ls=lstyle, alpha=alpha, marker=marker, ms=0.5, label=label)

    for i,r in enumerate(ill_avg):
        ax2.plot(r['t'], r['time_interval'], color=col, lw=lw, ls=lstyle, marker=marker, ms=0.5, alpha=alpha)

    for i,r in enumerate(ili_avg):
        ax3.plot(r['t'], r['time_interval'], color=col, lw=lw, ls=lstyle, marker=marker, ms=0.5, alpha=alpha)

    ax1.set_title("Time interval IS-L")
    ax2.set_title("Generation time L-L")
    ax3.set_title("Time interval IL_I")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.set_xlim(0,xlim)
    ax2.set_xlim(0,xlim)
    ax3.set_xlim(0,xlim)
    ax1.set_ylim(0,7)
    ax2.set_ylim(0,7)
    ax3.set_ylim(0,10)
    #ax3.set_title("Time interval pot_real IS-L")

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
        xlim = 40.
        marker = global_dict['marker']
        ax.plot(times, infected/N, color=col, lw=lw, marker=marker, ms=0.5, markevery=10, alpha=alpha) #, label=f"{vacc1_rate}") 
        ax.plot(times, recov/N, color=col, lw=lw, marker=marker, ms=0.5, markevery=10, alpha=alpha) #, label="R")
        print("times= ", times)
        quit()
        ax.grid(True)
        ax.set_xlim(0,xlim)
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
    axv.legend(title="R0, beta_shape$, beta_scale$", loc='upper right')
    axv.grid(True)
    axv.set_title("Impact of infectivity profile")
    axv.set_xlabel("Time")
    axv.set_xlim(0,xlim)

def plot_latent_by_degree(delta_time, ax, folders, global_dict):
    if 1:
        alpha = 0.7
        lw = 0.5
        col = global_dict['color']
        N = global_dict['N']  
        run = global_dict['run']
        marker = global_dict['marker']
        xlim = 40.
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
        ax.plot(times, latent/N, color=col, lw=lw, marker=marker, ms=0.5, markevery=10, alpha=alpha)
    ax.set_xlim(0,xlim)
    ax.set_title("Fraction Latent")
    ax.grid(True)


#----------------------------------------------------------------------
if __name__ == "__main__":

    # This script presumes a loop over a degrees list (hardcoded in the various methods)

    # one color per case
    cols = ['r', 'g', 'b', 'c', 'm', 'orange', 'black']

    # Store the plots in variables. Rearrange later if possible.

    dt = 0.1  # time step in units of days
    r, c = 4,2
    fig, axes = plt.subplots(r, c, figsize=(12,10))
    #axes = np.asarray(axes).reshape(-1)

    handles_R = {}
    hd = {}; hdc = {}

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
          if global_dict['beta_scale'] == 3.0: global_dict['color'] = 'r'
          if global_dict['beta_scale'] == 5.0: global_dict['color'] = 'g'
          if global_dict['beta_scale'] == 7.0: global_dict['color'] = 'b'
          if global_dict['beta_shape'] == 2.0: global_dict['ltype'] = '-'
          if global_dict['beta_shape'] == 5.0: global_dict['ltype'] = '-.'
          if global_dict['R0'] == 2.0: global_dict['marker'] = 'o'
          if global_dict['R0'] == 2.5: global_dict['marker'] = '^'
          if global_dict['R0'] == 3.0: global_dict['marker'] = '>'
        except:
          global_dict['color'] = 'b'
        global_dict['run'] = run

        # Individual R and counts of individuals infected (symptomatic) each day (incidence)
        h, hc = plot_individual_R(axes[0,0], axes[1,0], folder, global_dict)
        hd[global_dict['top_level_run']] = h
        hdc[global_dict['top_level_run']] = hc

        plot_infections_by_degree(dt, axes[0,1], axes[1,1], folder, global_dict)
        plot_latent_by_degree(dt, axes[2,0], folder, global_dict)
        plot_generation_times_2(axes[2,1], axes[3,0], axes[3,1], folder, global_dict)

    hd  = list(hd.values()); hdc = list(hdc.values())
    fsz = 8
    leg_text = "R0, beta_shape, beta_scale"
    lg = axes[0,0].legend(handles=hd, title=leg_text, loc='upper right', ncol=1, fontsize=fsz)
    lgc = axes[1,0].legend(handles=hdc, title=leg_text, loc='upper right', ncol=1, fontsize=fsz)
    #print(help(lg.set_title))
    lg.get_title().set_fontsize(str(fsz))
    lgc.get_title().set_fontsize(str(fsz))
    plt.tight_layout()
    plt.savefig(f'plot_multis_project{case}.pdf')

