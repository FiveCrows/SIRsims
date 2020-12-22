
import pandas as pd
import glob
import os, sys, shutil
import stats
from scanf import scanf

L  = 1
IS = 4
R  = 8

all_files = glob.glob("graphs/*")
folders = []

dirs = list(os.scandir("graphs"))
rows = []
run  = 0

for source_folder in list(dirs):
    if os.path.isfile(source_folder.path): continue
    #print(source_folder.path)
    strg, degree = scanf("%s_%d", source_folder.path) 
    dest_folder = "r_seir" + "/" + source_folder.path
    filenm = dest_folder + "/" + "transition_stats.csv"
    try:
        df, IS_L, IS_R, L_IS = stats.getDataframe(filenm)
        nb_rows = df.shape[0]
    except:
        continue

    print("-----------------------------------------------")
    print("         %s (%d rows)" % (dest_folder, nb_rows))

    label, mean, var = stats.processTransmissionTimes(L_IS, "L_IS", plot_data=False)
    label, mean, var = stats.processTransmissionTimes(IS_R, "IS_R", plot_data=False)
    label, mean, var = stats.processTransmissionTimes(IS_L, "IS_L", plot_data=False)

    rows.append([run, degree, label, mean, var])
    #if (run > 5): break
    run += 1

df = pd.DataFrame(rows)
df.columns = ["run", "degree", "label", "mean", "var"]
df.to_csv("degrees_mean_var.csv", index=False)
