# Each subplot contains normalized infection curves for either
# - different percentages of vaccination keeping 5000 top workplaces
# - different number of workplaces assuming 75% vaccination

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


#-----------------------------------------------------
df = pd.read_pickle("transformed_metadata.gz")

df_glob = df.global_dict.apply(pd.Series)
df = df.drop("global_dict", axis=1)
#df1 = df.join(df_glob)
dfg = pd.concat([df, df_glob], axis=1)    # Double indexing
#print(dfg.columns); quit()
# Rename columns
ren = { 'loop_nb_wk' : 'nb_wk',
       "loop_nb_sch" : 'nb_sch',
       "loop_v_pop_perc" : 'v_pop_perc',
       "loop_perc_vacc" : 'perc_vacc', # fraction vaccinated at work
      }
dfg.rename(columns=ren, inplace=True)

# We study each of the loops
# plot I curves for each case

#print(dfg["ages_SIR"]); quit()
#print(dfg["ages"]); quit()  # (the same thing)
df_nbwk   = dfg.groupby("ages_SIR")
df_perc_vacc   = dfg.groupby("perc_vacc")
# doing a mean removed all columns for which a mean was not possible, 
# such as dir curves, etc. I would like to do a mean over SIR curves for 
# multiple simulations to compute an average. But the lengths might all be
# the same. But I could take the first 100 values. The means would have to be 
# be computed manually since the data is stored as dictionaries of lists
#df_perc_nbwrk = dfg.groupby(["nb_wk", "perc_vacc"]).mean()

# extract all records with perc_vacc=75% and nb_wk=5000
#df1 = df[(df['perc_vacc'] == 0.75) & (df['nb_wk'] == 5000)]
print(df.columns)



df1 = dfg[(dfg['perc_vacc'] == 0.75)]
df1 = dfg[(dfg['perc_vacc'] == 0.75) & (dfg['nb_wk'] == 5000)]
df_perc_vacc = dfg[(dfg['nb_wk'] == 5000)]
print(df1['perc_vacc'])
print(df1['nb_wk'])

#----------------------------------------
def plotAgesSubgraphs(df):
    df_perc_nbwrk = df.groupby(["nb_wk"])
    keys = list(df_perc_nbwrk.groups.keys())
    nrow, ncol = u.nbSubplots(20)  # nb ages
    plt.subplots(nrow, ncol)

    row = df.iloc[0]
    #print(df.columns)
    #ages = row.ages_SIR # keys is time
    #print(ages.keys())

    ages = row.SIR_by_age  # keys is age
    print(ages.keys())
    plt.suptitle("Infection curves for each age bracket\n\
         As a function of the vaccination % in the 5000 top workplaces", fontsize=6)

    for row in df.itertuples():
      for age_key in range(20):
        plt.subplot(nrow, ncol, age_key+1)
        sir = row.SIR_by_age[age_key]
        N = sir['I'][0] + sir['S'][0] + sir['R'][0]
        if N == 0: continue
        I = sir['I'] / N
        plt.plot(sir['t'], I, lw=0.5, label="%d" % int(100*row.perc_vacc))
        plt.ylim(0., 0.4)
        age_min, age_max = 5*age_key, 5*(age_key+1)-1
        plt.title("age bracket: %d-%d" % (age_min, age_max))
        plt.legend(title="% vacc", fontsize=3)

    plt.tight_layout()
    plt.savefig("gordon.pdf")
    quit()

    
    for k, key in enumerate(df_perc_nbwrk.groups.keys()):
        dfk = df_perc_nbwrk.get_group(key)  # dataframe
        dfk = dfk.sort_values(by=['perc_vacc'])
        plt.subplot(nrow, ncol, k+1)
        plt.suptitle("Vaccinations in the Workplace")

        for idx, r in enumerate(dfk.itertuples()):
            sir = r.SIR
            N = sir['I'][0] + sir['S'][0] + sir['R'][0]
            plt.plot(sir['t'], sir['I']/N, lw=0.1, label="%d%%" % int(100*r.perc_vacc))
            plt.ylim(0, 0.3)
            plt.title("#wk: %d" % (r.nb_wk), size=6)
        plt.legend(fontsize=4)
    
    plt.tight_layout()
    plt.savefig("workplace_vacc_1.pdf")

#----------------------------------------
plotAgesSubgraphs(df_perc_vacc)
