import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rc('font', size=12)
matplotlib.rc('xtick', labelsize=10) # axis tick labels
matplotlib.rc('ytick', labelsize=10) # axis tick labels
matplotlib.rc('axes', labelsize=10)  # axis label
matplotlib.rc('axes', titlesize=10)  # subplot title
matplotlib.rc('figure', titlesize=10)

# Take vaccines into account. 
# reduction is 0 in all cases. 
# So there are two levels of variation: 
#  1. vaccination percentage of the population 
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

plt.title("Global Infection\n50% population masked and social distancing\n50% reduction of both")

curves = []

for i,r in enumerate(df.itertuples()):
        sir = r.SIR
        if r.sm != 0.5: continue
        vac = r.vaccination_dict
        nb_top_wk = vac['nb_top_workplaces_vaccinated']
        nb_top_sch = vac['nb_top_schools_vaccinated']
        nb_wk_vac = vac['workplace_population_vaccinated']
        nb_sch_vac = vac['school_population_vaccinated']
        N = sir['I'][0] + sir['S'][0] + sir['R'][0]
        frac_pop_vac = (nb_wk_vac+nb_sch_vac) / N
        I = np.asarray(sir['I']) / N
        nb_wk = vac['nb_workplaces']
        nb_sch = vac['nb_schools']
        curves.append([nb_top_wk, sir['t'], I, r.v, frac_pop_vac, nb_wk, nb_top_sch, nb_sch_vac])
        print("nb_top_wk= ", nb_top_wk)
        print("nb_wk_vac, nb_sch_vac= ", nb_wk_vac, nb_sch_vac)
        print(vac)
        print("N= ", N)
        print("Initial S, I, R: ", sir['S'][0], sir['I'][0], sir['R'][0])
        print("nb_top_wk= ", nb_top_wk)

curves = sorted(curves, key=lambda x: x[6])

for c in curves:
        v = c[3]
        t = c[1]
        I = c[2]
        frac_pop_vac = c[4]
        nb_top_wk = c[0]
        nb_wk = c[5]
        nb_top_sch = c[6]
        nb_sch_vac = c[7]
        frac_work = nb_top_wk / nb_wk
        frac_school = nb_top_sch / nb_sch

        if v > 0.95:
            plt.plot(t, I, "-", label="%3.2f, %3.2f, %4d, %3.2f" % (v, frac_pop_vac, nb_top_sch, frac_school))
        else:
            plt.plot(t, I, label="%3.2f, %3.2f, %4d, %3.2f" % (v, frac_pop_vac, nb_top_sch, frac_school))

        plt.xlabel("Time")
        plt.ylabel("Normalized Infections")
        plt.xlim(0, 100)
        #plt.ylim(0, 0.4)
        plt.legend(fontsize=6)
        #print(i, r.sm, r.v)

print(df.columns)
plt.tight_layout()
plt.savefig("global_infections_vaccpec.pdf")
quit()


