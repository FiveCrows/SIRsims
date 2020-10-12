import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as u

df = pd.read_pickle("metadata.gz")
df.to_csv("df.csv")

# Choose all elemnets with 
# DataFrame headers
# ,sm,sd,wm,wd,red_mask,red_dist,tau,gamma,SIR,fn,ages
# Choose frames with sm = sd = 1, wm = wd = 0

#dff = df[(df.sm == 1) & (df.sd == 1) & (df.wm == 0) & (df.wd == 0), :]
dfg = df.groupby(['sm','sd','wm','wd'])
dfs = {}
dfs[(1,1,0,0)] = dfg.get_group((1,1,0,0))
dfs[(1,0,0,0)] = dfg.get_group((1,0,0,0))
dfs[(0,1,0,0)] = dfg.get_group((0,1,0,0))
dfs[(0,0,0,0)] = dfg.get_group((0,0,0,0))
dfs[(1,1,1,1)] = dfg.get_group((1,1,1,1))

# Plot SIR curves per age as a function of time

#sir = row.SIR
#S, I, R, t = sir['S'], sir['I'], sir['R'], sir['t']
#print("filename: ", row.fn)

#S0 = S[0]
#plt.plot(t, np.asarray(S)/S[0])
#plt.plot(t, np.asarray(I)/S[0])
#plt.plot(t, np.array(R)/S[0])
#plt.show()

#---------------------
# Choose rows according to some criteria from the pandas table
# For example, wm=1, wd=1, sm=0, sd=0
# and then all the reductions of masks and keeping social reductions constant. 
# or create a 2D plot of max infection as a function of age somehow
#---------------------

def plotInfections1(age_brackets, df0, row_list):
    df = df0[df0['red_mask'] == 0.1]
    row_list = range(0,5)

    for r in row_list:
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

    for r in row_list:
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

