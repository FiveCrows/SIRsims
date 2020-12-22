import glob
import os, sys, shutil

folders = []

dirs = list(os.scandir("graphs"))
print(dirs)

nb_to_run = 10
count = 0

processed_dirs = []

for source_folder in list(dirs):
    if count > nb_to_run: break
    if os.path.isfile(source_folder.path):
        print("%s, Not a folder" % source_folder.path)
        continue
    processed_dirs.append(source_folder.path)
    count += 1
    dest_folder = "r_seir" + "/" + source_folder.path
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy("r_seir/parameters_0.txt", dest_folder)
    cmd = "./seir %s %s > output.txt" % (source_folder.path, dest_folder)
    print("-----------------------------------------------")
    print(cmd)
    os.system(cmd)
    shutil.copy("output.txt", dest_folder)
    shutil.copy("transition_stats.csv", dest_folder)
    os.remove("output.txt")
    os.remove("transition_stats.csv")

print("-----------------------------------------------")
print("Processed: ", processed_dirs)
print("Created output.txt and transition_stats.csv in subfolders")
