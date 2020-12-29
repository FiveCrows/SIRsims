import numpy as np
import glob
import os, sys, shutil
import stats
import matplotlib.pyplot as plt
import read_data as rd

L  = 1
IS = 4
R  = 8
PotL  = 10


def compute_stats():
    for source_folder in list(dirs):
        dest_folder = "r_seir" + "/" + source_folder.path
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

def plot_individual_R(ax):
    # Daily average individual reproduction number
    degrees = [5, 10, 20]
    ravg_l = []
    for degree in degrees:
        filenm = glob.glob('r_seir/graphs/*_%d/transition_stats.csv'%degree)[0]
        df, IS_L, IS_R, L_IS, IS_PotL = stats.getDataframe(filenm)
        _,_,_, df_Ravg    = stats.processTransmissionTimesInTime(IS_L, "IS_L", plot_data=False)
        _,_,_, df_RavgPot = stats.processTransmissionTimesInTime(IS_PotL, "IS_PotL", plot_data=False)
        ravg_l.append({'deg':degree, 't': df_Ravg['from_day'], 'R': df_Ravg['avgR'], \
                'tpot': df_RavgPot['from_day'], 'Rpot': df_RavgPot['avgR']})
    
    cols = ['r', 'g', 'b']
    for i, r in enumerate(ravg_l):
        if degrees[i] != 5: continue
        c = cols[i]
        ax.plot(r['t'], r['R'], '-', color=c, lw=1, label=r['deg'])
        ax.plot(r['tpot'], r['Rpot'], '.', color=c, lw=1, markersize=2, label=r['deg'])
    for z in zip(r['R'], r['Rpot']):
        print(z)
    for z in zip(r['t'], r['tpot']):
        print(z)
    ax.set_xlabel('Days')
    ax.set_ylabel('Individual R')
    ax.set_title('Individual R for Configuration model\nFixed degree')
    #plt.subtitle('(need to plot the degree histogram as well)')
    ax.legend(title='degree')

def plot_generation_times(df, ax1, ax2, ax3):
    degrees = [5, 10, 20]
    isl_avg = []
    isr_avg = []
    ispot_avg = []
    for degree in degrees:
        filenm = glob.glob('r_seir/graphs/*_%d/transition_stats.csv'%degree)[0]
        df, IS_L, IS_R, L_IS, IS_PotL = stats.getDataframe(filenm)
        # get average time intervals by day
        isl = IS_L.copy()
        isl['time_interval'] = isl['to_time'] - isl['from_time']
        isl['from_day'] = [int(i) for i in isl['from_time']]
        mti = mean_time_intervals = isl.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        isl_avg.append({'deg':degree, 't': mti['from_day'], 'time_interval' : mti['time_interval']})

        isr = IS_R.copy()
        isr['time_interval'] = isr['to_time'] - isr['from_time']
        isr['from_day'] = [int(i) for i in isr['from_time']]
        mti = mean_time_intervals = isr.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        isr_avg.append({'deg':degree, 't': mti['from_day'], 'time_interval' : mti['time_interval']})

        df_pot = IS_L.copy()
        df_pot = df_pot.append(IS_PotL)
        df_pot['time_interval'] = df_pot['to_time'] - df_pot['from_time']
        df_pot['from_day'] = [int(i) for i in df_pot['from_time']]
        mti = mean_time_intervals = df_pot.groupby('from_day').agg({'time_interval':'mean'}).reset_index()
        ispot_avg.append({'deg':degree, 't': mti['from_day'], 'time_interval' : mti['time_interval']})


    cols = ['r', 'g', 'b']
    for i,r in enumerate(isl_avg):
        ax1.plot(r['t'], r['time_interval'], color=cols[i], lw=1, label=r['deg'])

    for i,r in enumerate(isr_avg):
        ax2.plot(r['t'], r['time_interval'], color=cols[i], lw=1, label=r['deg'])

    for i,r in enumerate(ispot_avg):
        ax3.plot(r['t'], r['time_interval'], color=cols[i], lw=1, label=r['deg'])

    ax1.legend(title='degree')
    ax2.legend(title='degree')
    ax3.legend(title='degree')
    ax1.set_title("Time interval IS-L")
    ax2.set_title("Time interval IS-R")
    ax3.set_title("Time interval pot_real IS-L")

def plot_infections_by_degree(delta_time, ax):
    degrees = [5, 10, 20]
    cols = ['r', 'g', 'b']
    for i, degree in enumerate(degrees):
        filenm = glob.glob('r_seir/graphs/*_%d/data_baseline_p0.txt'%degree)[0]
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run = 0
        latent, infected, recov = rd.get_SEIR(by, run, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(infected)))
        ax.plot(times, infected, color=cols[i], label=degree)
    ax.legend(title='degree')
    ax.set_title("Nb Infected")

def plot_latent_by_degree(delta_time, ax):
    degrees = [5, 10, 20]
    cols = ['r', 'g', 'b']
    for i, degree in enumerate(degrees):
        filenm = glob.glob('r_seir/graphs/*_%d//data_baseline_p0.txt'%degree)[0]
        df = rd.setupDataframe(filenm)
        by = df.groupby("run")
        run = 0
        latent, infected, recov = rd.get_SEIR(by, run, delta_t=0.1, plot_data=False)
        times = delta_time * np.asarray(range(len(latent)))
        ax.plot(times, latent, color=cols[i], label=degree)
    ax.legend(title='degree')
    ax.set_title("Nb Latent")


#----------------------------------------------------------------------
if __name__ == "__main__":

    folders = []
    dirs = list(os.scandir("graphs"))
    
    #plt.figure(figsize=(8, 6))
    dt = 0.1  # time step in units of days
    r, c = 3,2
    fig, axes = plt.subplots(r, c, figsize=(10,10))
    axes = np.asarray(axes)
    print(axes)

    plot_individual_R(axes[0,0])
    plot_infections_by_degree(dt, axes[0,1])
    plot_latent_by_degree(dt, axes[1,0])
    plot_generation_times('is_l', axes[2,0], axes[2,1], axes[1,1])

    plt.tight_layout()
    plt.savefig('plot_individual_R.pdf')

    
