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
        plt.plot(c.t, c.I, "-", lw=.1)
        plt.xlabel("Time")
        plt.ylabel("I/N")
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
ren = { 'loop_nb_wk' : 'nb_wk',
       "loop_nb_sch" : 'nb_sch',
       "loop_v_pop_perc" : 'v_pop_perc',
      }
dfg.rename(columns=ren, inplace=True)

# We study each of the loops

# plot I curves for each case

df_nbwk   = dfg.groupby("nb_wk")
key1 = list(df_nbwk.groups.keys())[2]
df_nbsch   = dfg.groupby("nb_sch")
df_vpop_perc = dfg.groupby("v_pop_perc")


def nbSubplots(nb_keys):
    if   nb_keys <= 4:  size = (2,2)
    elif nb_keys <= 6:  size = (2,3)
    elif nb_keys <= 9:  size = (3,3)
    elif nb_keys <= 12: size = (3,4)
    return size

def plot_df(df_groupby, name):
    # df_groupby: is not a dataframe, 
    #      but a dataframe.groupby object

    nb_keys = len(df_groupby.groups.keys())
    nrow, ncol = nbSubplots(nb_keys)

    plt.subplots(nrow, ncol)
    plt.suptitle("%s" % "Infections as a function of fraction of businesses vaccinated\n\
           k is the fraction of largest businesses vaccinated")
    print("nb keys: ", df_groupby.groups.keys())

    for ix,k in enumerate(df_groupby.groups.keys()):
        print("ix= ", ix)
        plt.subplot(nrow, ncol, ix+1)
        df1 = df_groupby.get_group(k)
        plotMasks_I_wrt_cv(df1, title=k)
        plt.title("k=%3.1f" % k)

    plt.tight_layout()
    plt.savefig("plot_%s_sir.pdf" % (name))

#----------------------------------------------
plot_df(df_nbwk, "nb_wk")

quit()

