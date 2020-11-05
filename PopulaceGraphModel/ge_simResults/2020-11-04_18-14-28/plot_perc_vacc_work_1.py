# For each value of per_vacc_work (fraction of people vaccinated in the workplace at t=0,
# plot the infection curves. 

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

matplotlib.rc('font', size=6)
matplotlib.rc('xtick', labelsize=4) # axis tick labels
matplotlib.rc('ytick', labelsize=4) # axis tick labels
matplotlib.rc('axes', labelsize=4)  # axis label
matplotlib.rc('axes', titlesize=4)  # subplot title
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
def nbSubplots(nb_keys):
    if   nb_keys <= 4:  size = (2,2)
    elif nb_keys <= 6:  size = (2,3)
    elif nb_keys <= 9:  size = (3,3)
    elif nb_keys <= 12: size = (3,4)
    elif nb_keys <= 20: size = (4,5)
    elif nb_keys <= 25: size = (5,5)
    elif nb_keys <= 36: size = (6,6)
    else: size = (7,7)
    return size
#-----------------------------------------------------
df = pd.read_pickle("transformed_metadata.gz")

df_glob = df.global_dict.apply(pd.Series)
df = df.drop("global_dict", axis=1)
#df1 = df.join(df_glob)
dfg = pd.concat([df, df_glob], axis=1)    # Double indexing
# Rename columns
ren = { 'loop_nb_wk' : 'nb_wk',
       "loop_nb_sch" : 'nb_sch',
       "loop_v_pop_perc" : 'v_pop_perc',
       "loop_perc_vacc" : 'perc_vacc', # fraction vaccinated at work
      }
dfg.rename(columns=ren, inplace=True)
print(dfg.columns); quit()

# We study each of the loops

# plot I curves for each case

df_nbwk   = dfg.groupby("nb_wk")
df_perc_vacc   = dfg.groupby("perc_vacc")
# doing a mean removed all columns for which a mean was not possible, 
# such as dir curves, etc. I would like to do a mean over SIR curves for 
# multiple simulations to compute an average. But the lengths might all be
# the same. But I could take the first 100 values. The means would have to be 
# be computed manually since the data is stored as dictionaries of lists
#df_perc_nbwrk = dfg.groupby(["nb_wk", "perc_vacc"]).mean()

#----------------------------------------
#---------------------------------------
def plotPerSubgraph_1():
    df_perc_nbwrk = dfg.groupby(["nb_wk"])
    keys = list(df_perc_nbwrk.groups.keys())
    nrow, ncol = nbSubplots(len(keys))
    plt.subplots(nrow, ncol)
    
    for k, key in enumerate(df_perc_nbwrk.groups.keys()):
        dfk = df_perc_nbwrk.get_group(key)  # dataframe
        plt.subplot(nrow, ncol, k+1)
        plt.suptitle("Vaccinations in the Workplace")
    
        for idx, r in enumerate(dfk.itertuples()):
            #print(r.vacc_dict)
            #print("nb_top: ", r.vacc_dict['nb_top_workplaces_vaccinated'])
            #print("perc_people_vaccinated_in_workplaces", r.vacc_dict['perc_people_vaccinated_in_workplaces'])
            #print("perc_workplace_vaccinated", r.vacc_dict['perc_workplace_vaccinated'])
            sir = r.SIR
            N = sir['I'][0] + sir['S'][0] + sir['R'][0]
            plt.plot(sir['t'], sir['I']/N, lw=0.1, label="%d%%" % int(100*r.perc_vacc))
            plt.ylim(0, 0.3)
            plt.title("#wk: %d" % (r.nb_wk), size=6)
        plt.legend(fontsize=4)
    
    plt.tight_layout()
    plt.savefig("gordon.pdf")
#----------------------------------------
#----------------------------------------
def plotPerSubgraph():
    df_perc_nbwrk = dfg.groupby(["nb_wk", "perc_vacc"])
    keys = list(df_perc_nbwrk.groups.keys())
    nrow, ncol = nbSubplots(len(keys))
    plt.subplots(nrow, ncol)
    
    for k, key in enumerate(df_perc_nbwrk.groups.keys()):
        dfk = df_perc_nbwrk.get_group(key)  # dataframe
        plt.subplot(nrow, ncol, k+1)
        plt.suptitle("junk")
    
        for idx, r in enumerate(dfk.itertuples()):
            #print(r.vacc_dict)
            print("nb_top: ", r.vacc_dict['nb_top_workplaces_vaccinated'])
            print("perc_people_vaccinated_in_workplaces", r.vacc_dict['perc_people_vaccinated_in_workplaces'])
            print("perc_workplace_vaccinated", r.vacc_dict['perc_workplace_vaccinated'])
            sir = r.SIR
            N = sir['I'][0] + sir['S'][0] + sir['R'][0]
            plt.plot(sir['t'], sir['I']/N, lw=0.1, label="%f2.1" % r.perc_vacc)
            plt.ylim(0, 0.3)
            plt.title("#wk: %d, %%vacc_wk: %3.2f" % (r.nb_wk, r.perc_vacc), size=4)
    
    plt.legend(fontsize=4)
    plt.tight_layout()
    plt.savefig("gordon.pdf")
#----------------------------------------
#plotPerSubgraph()
plotPerSubgraph_1()
