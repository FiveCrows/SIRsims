import glob
import os, sys, shutil
import read_parameter_file as rpf
import pandas as pd

from datetime import datetime
timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

folders = []
global_dict = {}

dirs = list(os.scandir("graphs"))
dirs = ["data_ge"]
print(dirs)

source_folder = "data_ge/"
dest_folder = source_folder + "/results/" 
script_file = "run_leon_simulations.py"
param_file = "data_ge/parameters_0.txt"
state_transition_file = "transition_stats.csv"
output_file = "output.txt"
print("output_file= ", output_file)

global_dict["dest_folder"]   = dest_folder
global_dict["output_file"]   = output_file
global_dict["source_folder"] = source_folder
global_dict["script_file"]   = script_file
global_dict["timestamp"]     = timestamp
global_dict["param_file"]    = param_file
global_dict["state_transition_file"] = state_transition_file

def run_simulation(global_dict):
    output_file = global_dict["output_file"]
    os.makedirs(dest_folder, exist_ok=True)
    rpf.readParamFile(param_file, global_dict)
    shutil.copy(param_file, dest_folder)
    shutil.copy("vaccines.csv", source_folder)  # might not exist
    cmd = "./seir %s %s >& %s" % (source_folder, dest_folder, output_file)
    global_dict[cmd] = "./seir %s %s >& %s" % (source_folder, dest_folder, output_file)
    print("-----------------------------------------------")
    print(cmd)
    os.system(cmd)
    shutil.copy(output_file, dest_folder)
    state_transition_file = global_dict["state_transition_file"]
    shutil.copy(state_transition_file, dest_folder)
    os.remove(output_file)
    os.remove(state_transition_file)

if __name__ == "__main__":
    run_simulation(global_dict)
    print(global_dict)
