import numpy as np
import glob
import os, sys, shutil
import matplotlib.pyplot as plt

# User libraries
import stats 
import read_data as rd

# Run this file on a single simulation output

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

def plot_individual_R(ax, folders):
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

def plot_generation_times(df, ax1, ax2, ax3, folders):
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

def plot_infections_by_degree(delta_time, ax, folders):
    cols = ['r', 'g', 'b']
    if 1:
        filenm = glob.glob(folders + "/data_baseline_p0.txt")[0]
        print("filenm= ", filenm)
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run = 0
        latent, infected, recov = rd.get_SEIR(by, run, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(infected)))
        N = 260542  # hardcoded. BAD
        ax.plot(times, infected/N, label="IS")
        ax.plot(times, latent/N, label="Latent")
        ax.plot(times, (infected+latent)/N, label="IS+L")
        ax.plot(times, recov/N, label="R")
    ax.legend()
    ax.set_title("Nb Infected / pop size")

def plot_latent_by_degree(delta_time, ax, folders):
    cols = ['r', 'g', 'b']
    if 1:
        filenm = glob.glob(folders + "/data_baseline_p0.txt")[0]
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run = 0
        latent, infected, recov = rd.get_SEIR(by, run, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(latent)))
        ax.plot(times, latent)
    ax.set_title("Nb Latent")


#----------------------------------------------------------------------
if __name__ == "__main__":

    # This script presumes a loop over a degrees list (hardcoded in the various methods)

    folders = []
    dirs = list(os.scandir("graphs"))
    
    #plt.figure(figsize=(8, 6))
    dt = 0.1  # time step in units of days
    r, c = 3,2
    fig, axes = plt.subplots(r, c, figsize=(10,10))
    axes = np.asarray(axes)
    print(axes)

    folders = 'data_ge/results'
    print("folders= ", folders)

    plot_individual_R(axes[0,0], folders)
    plot_infections_by_degree(dt, axes[0,1], folders)
    plot_latent_by_degree(dt, axes[1,0], folders)
    plot_generation_times('is_l', axes[2,0], axes[2,1], axes[1,1], folders)

    plt.tight_layout()
    plt.savefig('plot_individual_R.pdf')

    
