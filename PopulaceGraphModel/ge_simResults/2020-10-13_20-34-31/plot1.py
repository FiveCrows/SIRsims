import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u

# Creation of different types of plots. To do this, create different queries on the dataframe
# fixed sm,sd=0, wm,wd=1, vary red_mask for fixed values of red_dist
#                         vary red_dist for fixed values of red_mask
# fixed sm,sd=1, wm,wd=0, vary red_mask for fixed values of red_dist
#                         vary red_dist for fixed values of red_mask
# fixed sm,sd=1, wm,wd=1, vary red_mask for fixed values of red_dist
#                         vary red_dist for fixed values of red_mask
#
# Plot infectives (I do not track deaths)
# I could plot maximum number infected as a function of red_dist and red_mask
#      or plot maximum fraction of infected as a function of red_dist and red_mask
# I could plot time to epidemic containment as a function of red_dist and red_mask
#               Hot to define containment?  *****
#
#
# Currently, reduction is 0.2,0.1,0.2,0.3,0.4 for red_mask and red_dist
# Rerun simulations with reductions of 0.2,0.4,0.6,0.8,1.0 (1.0 to test algorithms)
#  If reduction is 1 (or 100%), this implies the mask or social distancing is 100% effective

df = pd.read_pickle("metadata.gz")
df.to_csv("df.csv")

# Choose all elemnets with 
# DataFrame headers
# ,sm,sd,wm,wd,red_mask,red_dist,tau,gamma,SIR,fn,ages
# Choose frames with sm = sd = 1, wm = wd = 0

#dff = df[(df.sm == 1) & (df.sd == 1) & (df.wm == 0) & (df.wd == 0), :]
dfg = df.groupby(['sm','sd','wm','wd'])
dfs = {}
dfs[(0.7,0.7,0.7,0.7)] = dfg.get_group((0.7,0.7,0.7,0.7))
#dfs[(0,0,0,0)] = dfg.get_group((0,0,0,0))

# Plot SIR curves per age as a function of time

#---------------------
# Choose rows according to some criteria from the pandas table
# For example, wm=1, wd=1, sm=0, sd=0
# and then all the reductions of masks and keeping social reductions constant. 
# or create a 2D plot of max infection as a function of age somehow
#---------------------

def plotInfections1(age_brackets, df0, row_list):
    df = df0[df0['red_mask'] == 0.1]
    row_list = range(0,5)

    for r in row_list: # one curve per row
        row = df.iloc[r]
        curves = u.generateCurves(row)

        for k in age_brackets:
            t = curves[k]['t']
            S0 = curves[k]['S'][0]
            #if k < 10:
            if 1:
                plt.plot(t, np.asarray(curves[k]['I'])/S0, label=(row['red_mask'], row['red_dist']))
            #else:
                #plt.plot(t, np.asarray(curves[k]['I'])/S0, linestyle="--", label=k)
            plt.title("bracket: %d" % k)
    plt.legend()

def plotInfections(age_brackets, df0, row_list):
    df = df0[df0['red_mask'] == 0.1]
    row_list = range(0,5)

    for r in row_list: # one curve per row
        row = df.iloc[r]
        curves = u.generateCurves(row)

        for k in age_brackets:
            t = curves[k]['t']
            S0 = curves[k]['S'][0]
            #if k < 10:
            if 1:
                plt.plot(t, np.asarray(curves[k]['I'])/S0, label=(row['red_mask'], row['red_dist']))
            #else:
                #plt.plot(t, np.asarray(curves[k]['I'])/S0, linestyle="--", label=k)
            plt.title("bracket: %d" % k)
    plt.legend()

# Plot a matrix of the maximum infected percentage as a function of reductions
def computeMaxInfections(age_bracket, df0):
    # I AM CALCULATING REDUNDANT STUFF. 
    k = age_bracket
    mat = np.zeros([6,6])   # <<<<
    mat_d = {}
    for r in df0.itertuples():  # loop over rows
        curves = u.generateCurves(r)
        red_mask = r.red_mask
        red_dist = r.red_dist
        mx = np.maximum(curves[k]['S'][0], 1)  # in case mx is zero
        mat_d[(red_mask,red_dist)] = np.max(curves[k]['I']) / mx

    # How to convert this information into a matrix
    red = np.linspace(0.,10,6) / 10.  # to avoid 0.3000004 <<<<< ERROR?
    for i,r1 in enumerate(red):
        for j,r2 in enumerate(red):
            try:
                print("r1,r2= ", r1, r2)
                mat[i,j] = mat_d[r1,r2]
                print("i,j,mat: ", i,j,r1,r2,mat[i,j])
            except:
                print("pass")
                pass
    return mat

def plotMaxInfections(df0, fn, sm, sd, wm, wd):
    fig, axes = plt.subplots(4,5, figsize=(10,8))
    fig.suptitle("Maximum infectivity by age category\nsm=%2.1f, sd=%2.1f, wm=%2.1f, wd=%2.1f" % (sm,sd,wm,wd))
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    axes = axes.reshape(-1)

    #0  2   4  6  8  10
    names = ['0', '.2', '.4', '.6', '.8', '1.']
    #ticks = [0., 0.5, 1.0]
    ticks = np.linspace(0,10,6) / 10.

    for k in range(20):
        if 1:
            mat = computeMaxInfections(k, df0)
            ax = axes[k]
            ax.xaxis.set_ticks_position('bottom')
            ax.imshow(mat, vmin=0.0, vmax=0.6, cmap='hot', extent=[-0.10,1.10,-0.10,1.10], origin='lower')
            #ax.imshow(mat, vmin=0.0, vmax=0.4, cmap='hot', extent=[-0.10,1.10,-0.10,1.10], interpolation='nearest', origin='lower')
            #ax.imshow(mat, cmap='hot', extent=[-0.05,1.05,-0.05,1.05], interpolation='nearest', origin='lower')
            nx,ny = len(names), len(names)  # not general
            for i in range(nx):
              for j in range(ny):
                  txt = "%4.2f" % mat[i,j]
                  if mat[i,j] > 0.30: col = 'black'
                  else: col = 'w'
                  text = ax.text(i*0.2, j*0.2, txt, ha="center", va="center", color=col, fontsize=4)

            ax.set_title("k=%d" % k)
            ax.set_xlabel("mask red.")
            ax.set_ylabel("distance red.")
            ax.set_xticklabels(names)
            ax.set_yticklabels(names)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(fn + ".pdf")
    
sm, sd, wm, wd = [0.7, 0.7, 0.7, 0.7]
plotMaxInfections(dfs[(sm,sd,wm,wd)], 'fil.7.7.7.7', sm, sd, wm, wd)
#plotMaxInfections(dfs[(0,0,0,0)], 'fil0000')
quit()
#-----------------------------------------------------
age = 4
plt.subplots(2,2)
plt.subplot(2,2,1)
plotInfections([age], dfs[(0,0,0,0)], range(0,25))
plt.subplot(2,2,2)
plotInfections([age], dfs[(0,1,0,0)], range(0,25))
plt.subplot(2,2,3)
plotInfections([age], dfs[(1,0,0,0)], range(0,25))
plt.subplot(2,2,4)
plotInfections([age], dfs[(1,1,0,0)], range(0,25))
plt.savefig("plot1.pdf")
print("save to plot1.pdf")


age = 4
plt.subplots(2,2)
plt.subplot(2,2,1)
plotInfections([age], dfs[(0,0,0,0)], range(0,25))
plt.subplot(2,2,2)
plotInfections([age], dfs[(1,1,1,1)], range(0,25))

age = 7
plt.subplot(2,2,3)
plotInfections([age], dfs[(0,0,0,0)], range(0,25))
plt.subplot(2,2,4)
plotInfections([age], dfs[(1,1,1,1)], range(0,25))
plt.savefig("plot2.pdf")
print("save to plot2.pdf")



quit()

