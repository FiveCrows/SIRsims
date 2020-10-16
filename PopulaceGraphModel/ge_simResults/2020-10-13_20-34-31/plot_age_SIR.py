import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# plot infection curves for age brackets. One age brack per subplot.
# 

# Return dictionary parameter/values of first row of a DataFrame
def getParams(dfs, params):
    row = dfs.head(1)[list(params)]
    dct = {}
    values = row.values[0]
    for i,p in enumerate(params):
        dct[p] = values[i]
    return dct

#-----------------------------------------------------
df = pd.read_pickle("transformed_metadata.gz")

# Choose all elements with
# DataFrame headers
# ,sm,sd,wm,wd,red_mask,red_dist,tau,gamma,SIR,fn,ages
# Choose frames with sm = sd = 1, wm = wd = 0

group_key = ['sm','sd','wm','wd']
dfg = df.groupby(group_key)

keys = []
dfs = {}
for key, gr in dfg:
    keys.append(key)
    dfs[key]= dfg.get_group(key)

#-----------------------------------------------------

#---------------------
# Choose rows according to some criteria from the pandas table
# For example, wm=1, wd=1, sm=0, sd=0
# and then all the reductions of masks and keeping social reductions constant. 
# or create a 2D plot of max infection as a function of age somehow
#---------------------

def computeAgeSIR(age_bracket, df0):
    k = age_bracket
    mat_d = {}

    for r in df0.itertuples():  # loop over rows
        curve = r.SIR_age[k]
        S = curve['S']
        I = curve['I']
        R = curve['R']
        t = curve['t']
        red_mask = r.red_mask
        red_dist = r.red_dist
        N_age = r.N_age[k]
        mat_d[(red_mask,red_dist)] = (S, I, R, t)

    return mat_d, N_age

#------------------------------------
# Plot age SIR curves for each age group for one choice of (red_mask, red_dist)
# and one choice of (sm,sd,wm,wd)
def plotAgeSIR(df0, dct, title):
    rows = df0[(df0["red_mask"] == 0.6) & (df0["red_dist"] == 0.6)]
    row = rows.iloc[0]

    title = "SIR by age, mask_red=%2.1f, dist_red=%2.1f\n" % (row.red_mask, row.red_dist)
    filnm = "ageSIR_redmask=%2.1f_redist=%2.1f," % (row.red_mask, row.red_dist)

    count = 0
    for k,v in dct.items():
        if count == 0:
            comma = ""
        else:
            comma = ","
        count += 1
        title += comma + " %s=%2.1f" % (k,v) 
        filnm += comma + "%s=%2.1f" % (k,v) 

    fig, axes = plt.subplots(4,5, figsize=(10,8))
    fig.suptitle(title)
    axes = axes.flatten()
    SIR = row.SIR_age
    N_age = row.N_age
    print("N_age= ", N_age)

    for k in range(0,19):  # age brackets 0-18
        ax = axes[k]
        # all ages have the same time range
        I,S,R,t = [SIR[k][l] for l in ['S','I','R','t']]
        ax.plot(t, S, 'r')
        ax.plot(t, I, 'g')
        ax.plot(t, R, 'b')
        age_str = str(5*k)+"-"+str(5*k+4)
        N =  N_age[k]
        ax.set_title(age_str+", N=%d"%N, fontsize=10)

    plt.tight_layout()
    plt.savefig(filnm+".pdf")
    quit()

#-------------------------------------------

# Plot I(t) for age category k for (red_dist,red_mask) = (0,.2,.4,.6,.8,1.)
# and one choice of (sm,sd,wm,wd)
def plotI_reduction(df0, dct, title):
    print(df0.columns)
    rows = df0[(df0["red_mask"] == 0.6) & (df0["red_dist"] == 0.6)]
    row = rows.iloc[0]

    title = "Infectivity by reduction (normalized age group pop size)\n" 
    filnm = "ageSIR_"

    count = 0
    for k,v in dct.items():
        if count == 0:
            comma = ""
        else:
            comma = ","
        count += 1
        title += comma + " %s=%2.1f" % (k,v) 
        filnm += comma + "%s=%2.1f" % (k,v) 

    fig, axes = plt.subplots(4,5, figsize=(10,8))
    fig.suptitle(title)
    axes = axes.flatten()

    reductions = [(r,r) for r in np.linspace(0, 10., 6)/10]
    N_age =  df0.iloc[0].N_age

    # Create empty plot with blank marker containing the extra label

    xticks = np.linspace(0,50,6).astype('int')
    xnames = np.vectorize(str)(xticks)
    yticks = np.linspace(0,5,6) / 10.
    ynames = np.vectorize(str)(yticks)

    for k in range(0,19):  # age brackets 0-18
        ax = axes[k]
        #ax.plot([], [], ' ', label="Reduction (%)")
        N_age_list = []
        for red in reductions:
            rm, rd = red
            rows = df0[(df0["red_mask"] == rm) & (df0["red_dist"] == rd)]
            row = rows.iloc[0]
            SIR = row.SIR_age
            N_age = row.N_age[k]
            N_age_list.append(N_age)
            print("red= ", red, ", N_age= ", row.N_age)
            S,I,R,t = [SIR[k][l] for l in ['S','I','R','t']]
            ax.plot(t, I/N_age, label="%3d"%(100*rm))
            ax.set_xlim(0,50)
            ax.set_ylim(0,.5)
            # Must call xticks before labels for proper positioning
            ax.set_xticks(xticks)
            ax.set_xticklabels(xnames, fontsize=6)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ynames, fontsize=6)
            ax.tick_params(axis='both', which='both', length=0)

        print("=== k= ", k)
        age_str = str(5*k)+"-"+str(5*k+4)
        print("N_age_list= ", N_age_list)
        ax.set_title(age_str+", N=%d" % N_age_list[0], fontsize=10)
        l = ax.legend(fontsize=6, framealpha=0.5, title="Reduction (%)")
        plt.setp(l.get_title(), fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(filnm+".pdf")
    quit()

#-----------------------------------------
def plotI_reduction_subcellsOnly(df0, dct, subcells, title):
    # Only plot Infection curves listed in subcells dictionary
    print(df0.columns)
    rows = df0[(df0["red_mask"] == 0.6) & (df0["red_dist"] == 0.6)]
    row = rows.iloc[0]

    title = "Infectivity by reduction (normalized age group pop size)\n" 
    filnm = "ageSIR_"

    count = 0
    for k,v in dct.items():
        if count == 0:
            comma = ""
        else:
            comma = ","
        count += 1
        title += comma + " %s=%2.1f" % (k,v) 
        filnm += comma + "%s=%2.1f" % (k,v) 

    fig, axes = plt.subplots(4,5, figsize=(10,8))
    fig.suptitle(title)
    axes = axes.flatten()

    reductions = [r for r in np.linspace(0, 10., 6)/10]
    N_age =  df0.iloc[0].N_age

    # Create empty plot with blank marker containing the extra label

    xticks = np.linspace(0,50,6).astype('int')
    xnames = np.vectorize(str)(xticks)
    #yticks = np.linspace(0,5,6) / 10.
    #ynames = np.vectorize(str)(yticks)

    for k in range(0,19):  # age brackets 0-18
        ax = axes[k]
        #ax.plot([], [], ' ', label="Reduction (%)")
        N_age_list = []
        I_to_plot = subcells[k]
        for i,j in I_to_plot:
            rm, rd = reductions[i], reductions[j]
            # if rm, rd not in the list to plot, skip
        #for red in reductions:
            rows = df0[(df0["red_mask"] == rm) & (df0["red_dist"] == rd)]
            row = rows.iloc[0]
            SIR = row.SIR_age
            N_age = row.N_age[k]
            N_age_list.append(N_age)
            S,I,R,t = [SIR[k][l] for l in ['S','I','R','t']]
            ax.plot(t, I/N_age, label="%3d"%(100*rm))
            ax.set_xlim(0,50)
            ax.set_ylim(0,.005)
            # Must call xticks before labels for proper positioning
            ax.set_xticks(xticks)
            ax.set_xticklabels(xnames, fontsize=6)
            #ax.set_yticks(yticks)
            #ax.set_yticklabels(ynames, fontsize=6)
            ax.tick_params(axis='both', which='both', length=0)

        print("=== k= ", k)
        age_str = str(5*k)+"-"+str(5*k+4)
        print("N_age_list= ", N_age_list)
        ax.set_title(age_str+", N=%d" % N_age_list[0], fontsize=10)
        l = ax.legend(fontsize=6, framealpha=0.5, title="Reduction (%)")
        plt.setp(l.get_title(), fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(filnm+".pdf")
    quit()

#-----------------------------------------
dfs0 = dfs[(.3,.3,.3,.3)]
dfs0 = dfs[(.7,.7,.7,.7)]
#-----------------------------------------
dfs0 = dfs[(.3,.3,.3,.3)]
dfs0 = dfs[(.7,.7,.7,.7)]
#plotAgeSIR(dfs0, dct, title="")
dct = getParams(dfs0, group_key)
#plotI_reduction(dfs0, dct, title="")
 

# Display the curves for these cases
ages = {}
ages[0] = [(5,4),(5,5)]
ages[1] = [(5,4),(5,5)]
ages[2] = [(5,5)]
ages[3] = [(5,4),(5,5)]
ages[4] = [(5,4),(5,5),(4,5)]
ages[5] = [(5,4),(5,5)]
ages[6] = [(5,4),(5,5)]
ages[7] = [(5,4),(5,5)]
ages[8] = [(5,4),(5,5)]
ages[9] = [(5,5)]
ages[10] = [(5,4),(5,5)]
ages[11] = [(5,4),(5,5),(4,5)]
ages[12] = [(5,4),(5,5),(4,5),(5,3)]
ages[13] = [(5,4),(5,5),(4,5)]
ages[14] = [(5,4),(5,5),(5,3)]
ages[15] = [(4,5),(5,3),(5,4),(5,5)]
ages[16] = [(5,1),(5,2),(5,3),(5,4),(5,5)]
ages[17] = [(5,4),(5,5),(4,5)]
ages[18] = [(5,0),(5,1),(5,3),(5,4),(5,5),(4,3),(4,4),(4,5),(3,4),(3,5),(2,5)]
ages_to_plot = ages

dfs0 = dfs[(.7,.7,.7,.7)]
dct = getParams(dfs0, group_key)
plotI_reduction_subcellsOnly(dfs0, dct, ages_to_plot, title="")
