import os, sys

# creates the full path
cmd = "mkdir -p data_ge/results"
os.system(cmd)
os.system("export LD_LIBRARY_PATH=/usr/local/lib")
if os.path.exists("data_ge/network.txt.gz"):
    os.system("cd data_ge; gunzip -c network.txt.gz > network.txt")

if os.path.exists("data_ge/nodes.txt.gz"):
    os.system("cd data_ge; gunzip -c nodes.txt.gz > nodes.txt")


os.system("make -f BA_makefile")
print("\n=========================================================")
print(  "===== make completed ====================================\n")

# Change the project number to 17 near the bottom of the file
os.system("python run_multi_leon_sims_1.py")
print("\n=============================================================")
print(  "===== run_multi_leon_sims_1.py script completed =============\n")

# Assume you ran in project 17
os.system("python plot_multi_generation_times.py 17")
print("\n==============================================================")
print(  "===== plot_multi_generation_times.py script completed ========\n")
