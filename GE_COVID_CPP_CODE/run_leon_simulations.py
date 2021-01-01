import glob
import os, sys, shutil

folders = []

dirs = list(os.scandir("graphs"))
dirs = ["data_ge"]
print(dirs)

source_folder = "data_ge/"


#for source_folder in list(dirs):
if 1:
    dest_folder = source_folder + "/results/" 
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy("data_ge/parameters_0.txt", dest_folder)
    shutil.copy("vaccines.csv", source_folder)  # might not exist
    cmd = "./seir %s %s >& output.txt" % (source_folder, dest_folder)
    #cmd = "./seir %s %s" % (source_folder, dest_folder)
    print("-----------------------------------------------")
    print(cmd)
    os.system(cmd)
    shutil.copy("output.txt", dest_folder)
    shutil.copy("transition_stats.csv", dest_folder)
    os.remove("output.txt")
    os.remove("transition_stats.csv")

print("Processed: ", dirs)
print("Created output.txt and transition_stats.csv in subfolders")
