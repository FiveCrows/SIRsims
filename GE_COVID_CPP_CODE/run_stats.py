
import glob
import os, sys, shutil
import stats

L  = 1
IS = 4
R  = 8

all_files = glob.glob("graphs/*")
folders = []

dirs = list(os.scandir("graphs"))

for source_folder in list(dirs):
    dest_folder = "r_seir" + "/" + source_folder.path
    filenm = dest_folder + "/" + "transition_stats.csv"
    try:
        df, IS_L, IS_R, L_IS = stats.getDataframe(filenm)
        nb_rows = df.shape[0]
    except:
        continue

    print("-----------------------------------------------")
    print("         %s (%d rows)" % (dest_folder, nb_rows))
    stats.processTransmissionTimes(L_IS, "L_IS", plot_data=False)
    stats.processTransmissionTimes(IS_R, "IS_R", plot_data=False)
    stats.processTransmissionTimes(IS_L, "IS_L", plot_data=False)
