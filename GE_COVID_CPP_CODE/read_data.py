
#### ERROR SOMETHING MISMATCH. WHY???  
#### Output files are empty. WHY? 

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# TO EXECUTE CODE
#  python read_data.py folder_name


def setupDataframe(filenm):
    text = np.loadtxt(filenm)
    df = pd.DataFrame(text)
    df.columns = [
        "l_asymp", 
        "l_sympt", 
        "i_asymp",
        "pre_sympt",
        "i_sympt",
        "home",
        "hospital",
        "icu",
        "recov",
        "new_l_asymp",
        "new_l_sympt",
        "new_i_asympt",
        "new_pre_sympt",
        "new_i_sympt",
        "new_home",
        "new_hostp",
        "new_icu",
        "new_recov",
        "run",]  # there are 100 runs
    return df


def plot_group(by, group):
    # Different groups have different lengths
    df = by.get_group(group)
    infected_s      = df["i_sympt"];
    pre_sympt       = df["pre_sympt"]
    lat_sympt       = df["l_sympt"]
    infected_total  = infected_s + lat_sympt + pre_sympt  
    recov           = df["recov"]

    names = [infected_s, pre_sympt, lat_sympt, recov, infected_total, infected_total]
    labels = ["infectious_s", "pre_s", "latent_s", "recovered", "total_infected"]
    colors = ["r", "orange", "b", "g", "black"]
    handles = []
    alpha = 0.8

    dt = 0.1  # might change

    for n,l,c in zip(names, labels, colors):
        times = dt * np.asarray(range(len(n)))
        p, = plt.plot(times, n, alpha=alpha, color=c, label=l)
        handles.append(p)
    return handles

if __name__ == "__main__":

    args = sys.argv[1:]
    folder = args[0]
    filenm = folder + "/data_baseline_p0.txt"
    df = setupDataframe(filenm)
    by = df.groupby("run")

    nb_runs = 1

    for run in range(0, nb_runs):
        handles = plot_group(by, run)

    plt.legend(handles=handles, loc='center right', ncol=1)
    plt.show()
    quit()

