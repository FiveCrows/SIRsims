import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rc('font', size=5)
matplotlib.rc('xtick', labelsize=6) # axis tick labels
matplotlib.rc('ytick', labelsize=6) # axis tick labels
matplotlib.rc('axes', labelsize=8)  # axis label
matplotlib.rc('axes', titlesize=10)  # subplot title
matplotlib.rc('figure', titlesize=8)

# Take vaccines into account. 
# reduction is 0 in all cases. 
# So there are two levels of variation: 
#  1. vaccination percentate of the population 
#  2. fraction of the population wearing masks and social distancing (same values)

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

fig, axes = plt.subplots(4,5, figsize=(10,6))
axes = axes.flatten()
fig.suptitle("Normalized Infection\n" + 
             "50% population masked and social distancing\n" + 
             "50% reduction of both, each school 100% vaccinated")

curves = []

for i,r in enumerate(df.itertuples()):
        sir = r.SIR
        if r.sm != 0.5: continue
        vac = r.vaccination_dict
        nb_top_wk  = vac['nb_top_workplaces_vaccinated']
        nb_top_sch = vac['nb_top_schools_vaccinated']
        nb_wk_vac  = vac['workplace_population_vaccinated']
        nb_sch_vac = vac['school_population_vaccinated']
        N = sir['I'][0] + sir['S'][0] + sir['R'][0]
        frac_pop_vac = (nb_wk_vac+nb_sch_vac) / N
        I            = np.asarray(sir['I']) / N
        nb_wk        = vac['nb_workplaces']
        nb_sch       = vac['nb_schools']
        curves.append([nb_top_wk, sir['t'], I, r.v, frac_pop_vac, nb_wk, nb_top_sch, nb_sch_vac, r.SIR_age, r.N_age])

curves = sorted(curves, key=lambda x: x[6])

for c in curves:
    v = c[3]
    t = c[1]
    I = c[2]
    nb_top_wk    = c[0]
    frac_pop_vac = c[4]
    nb_wk        = c[5]
    nb_top_sch   = c[6]
    nb_sch_vac   = c[7]
    sir_age      = c[8]
    N_age        = c[9]
  
    for age in range(19):
        ax = axes[age]
        sir = sir_age[age] 
        I = sir['I'] / N_age[age]
        ax.plot(sir['t'], I, label="%2d" % nb_top_sch)
        ax.set_title("k=%2d" % age, fontsize=6)
        ax.set_xlabel("Time", fontsize=4)
        ax.set_ylabel("Normalized Infections", fontsize=4)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 0.4)
        ax.legend(fontsize=6)
    print(i, r.sm, r.v)

print(df.columns)
plt.tight_layout()
plt.savefig("Infections_per_age_per_vaccpec.pdf")
quit()


