import os, sys

# creates the full path
cmd = "mkdir -p data_ge/results"
os.system(cmd)

if os.path.exists("data_ge/network.txt.gz"):
    os.system("cd data_ge; gunzip -c network.txt.gz > network.txt")

if os.path.exists("data_ge/nodes.txt.gz"):
    os.system("cd data_ge; gunzip -c nodes.txt.gz > nodes.txt")


os.system("make")
print("\n=========================================================")
print(  "===== make completed ====================================\n")

os.system("python run_multi_leon_sims_1.py")
print("\n=============================================================")
print(  "===== run_multi_leon_sims_1.py script completed =============\n")

os.system("python plot_multi_generation_times.py")
print("\n==============================================================")
print(  "===== plot_multi_generation_times.py script completed ========\n")
