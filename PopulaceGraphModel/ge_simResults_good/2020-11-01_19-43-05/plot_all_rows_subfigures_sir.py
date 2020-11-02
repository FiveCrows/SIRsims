# 
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib
from collections import namedtuple

matplotlib.rc('font', size=12)
matplotlib.rc('xtick', labelsize=10) # axis tick labels
matplotlib.rc('ytick', labelsize=10) # axis tick labels
matplotlib.rc('axes', labelsize=10)  # axis label
matplotlib.rc('axes', titlesize=10)  # subplot title
matplotlib.rc('figure', titlesize=10)

# Return dictionary parameter/values of first row of a DataFrame
def getParams(dfs, params):
    row = dfs.head(1)[list(params)]
    dct = {}
    values = row.values[0]
    for i,p in enumerate(params):
        dct[p] = values[i]
    return dct

#-----------------------------------------------------
def plotMasks_I_wrt_cv(df, title):
    curves = []
    C = namedtuple('C', 't I')
    for i,r in enumerate(df.itertuples()):
        sir = r.SIR
        N = sir['I'][0] + sir['S'][0] + sir['R'][0]
        I = np.asarray(sir['I']) / N
        curves.append(C(t=sir['t'], I=I))

    # Notice the use of the namedtuple for readability
    #curves = sorted(curves, key=lambda x: x.cv_val)

    for c in curves:
        plt.plot(c.t, c.I, "-")
        plt.xlabel("Time")
        plt.ylabel("Normalized Infections")
        plt.xlim(0, 150)
        #plt.legend(fontsize=6)
        plt.title(title)

#-----------------------------------------------------
df = pd.read_pickle("transformed_metadata.gz")

df_glob = df.global_dict.apply(pd.Series)
df = df.drop("global_dict", axis=1)
#df1 = df.join(df_glob)
dfg = pd.concat([df, df_glob], axis=1)
# Rename columns
ren = {'loop_avg_efficiency':'avg_eff',
       'loop_std_efficiency':'std_eff',
       'loop_adoption':'adopt',
       'loop_sim_repeat':'sim_rep',
      }
dfg.rename(columns=ren, inplace=True)

# We study each of the loops

# plot I curves for each case

df_avg   = dfg.groupby("avg_eff")
df_std   = dfg.groupby("std_eff")
df_adopt = dfg.groupby("adopt")

"""
print("len: ", len(df_avg.groups))
ix = list(df_avg.groups[0.5])
a = dfg.iloc[ix]['avg_eff']
keys = list(df_avg.groups.keys())
"""

def plot_df(df_groupby, name):
    # df_groupby: is not a dataframe, 
    #      but a dataframe.groupby object
    plt.subplots(2,2)
    plt.suptitle("%s" % name)

    for ix,k in enumerate(df_groupby.groups.keys()):
        plt.subplot(2,2,ix+1)
        df1 = df_groupby.get_group(k)
        plotMasks_I_wrt_cv(df1, title=k)
        plt.title("k=%f" % k)

    plt.tight_layout()
    plt.savefig("plot_%s_sir.pdf" % (name))

plot_df(df_avg, "avg")
plot_df(df_std, "std")
plot_df(df_adopt, "adoption")

quit()
embed()

print(df_avg); quit()
quit()


"""
Parameters varied: 
         glob_dict['loop_sim_repeat'] = count
         glob_dict["loop_avg_efficiency"] = avg_efficacy
         glob_dict["loop_std_efficiency"] = std_efficacy
         glob_dict["loop_adoption"] = adoption
#
# 3 simulations for each case
  for avg_efficacy in [0., 0.25, 0.5, 0.75]:
     for std_efficacy in [0., 0.3, 0.6]:
       for adoption in [0.0, 0.5, 1.]:  # masks and social distancing in schools and workplaes
"""

plotMasks_I_wrt_cv(df, '')

print(df.columns)
plt.tight_layout()
plt.savefig("plot_all_sir.pdf")


