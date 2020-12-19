import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

filenm = "data_baseline_p0.txt"

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

print(df)
by = df.groupby("run")

def plot_group(by, group):
    # Different groups have different lengths
    df = by.get_group(group)
    infected    = df["l_sympt"] + df["i_sympt"] + df["pre_sympt"] 
    plt.plot(range(len(infected)),infected, label="i")
    recov = df["recov"]
    plt.plot(range(len(recov)), recov, label="r")

for i in range(0,5):
    plot_group(by, i)
plt.show()
quit()

