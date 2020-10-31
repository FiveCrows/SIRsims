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
df = pd.read_pickle("transformed_metadata.gz")

plt.title("Global Infection\n50% population masked and social distancing\n50% reduction of both")

# Change 0.199999 to 0.20 for example
df['cv_val']    = round(df['cv_val'], 2)

# extract rows to study
df_mask    = df[df['cv_key'] == 'mask_eff']
df_contact = df[df['cv_key'] == 'contact']
df_weight  = df[df['cv_key'] == 'weight']

def plotMasks_I_wrt_cv(df, title):
    curves = []
    C = namedtuple('C', 't cv_val I')
    for i,r in enumerate(df.itertuples()):
        sir = r.SIR
        vac = r.vacc_dict
        N = sir['I'][0] + sir['S'][0] + sir['R'][0]
        I = np.asarray(sir['I']) / N
        curves.append(C(t=sir['t'], cv_val=r.cv_val, I=I))

    # Notice the use of the namedtuple for readability
    curves = sorted(curves, key=lambda x: x.cv_val)

    for c in curves:
        plt.plot(c.t, c.I, "-", label=c.cv_val)
        plt.xlabel("Time")
        plt.ylabel("Normalized Infections")
        plt.xlim(0, 150)
        plt.legend(fontsize=6)
        plt.title(title)

plt.subplots(2,2)
plt.subplot(2,2,1)
plotMasks_I_wrt_cv(df_mask, 'Masks')
plt.subplot(2,2,2)
plotMasks_I_wrt_cv(df_contact, 'Contacts')
plt.subplot(2,2,3)
plotMasks_I_wrt_cv(df_weight, 'Distancing')

print(df.columns)
plt.tight_layout()
plt.savefig("global_infections_Icurve.pdf")


