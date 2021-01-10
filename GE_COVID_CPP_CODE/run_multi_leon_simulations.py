import glob
import os, sys, shutil
import read_parameter_file as rpf
import pandas as pd
import traceback
import pickle

from datetime import datetime
timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

folders = []
global_dict = {}

#dirs = list(os.scandir("graphs"))
#dirs = ["data_ge"]
#print(dirs)

def setupGlobalDict():
    source_folder = "data_ge/"
    base_dest_folder = source_folder + "/results/"  # no final slash
    dest_folder = source_folder + "/run1/"  # no final slash
    script_file = "run_leon_simulations.py"
    param_file = "data_ge/parameters_0.txt"
    state_transition_file = "transition_stats.csv"
    output_file = "output.txt"
    
    global_dict["base_dest_folder"] = base_dest_folder
    global_dict["dest_folder"]   = dest_folder
    global_dict["output_file"]   = output_file
    global_dict["source_folder"] = source_folder
    global_dict["script_file"]   = script_file
    global_dict["timestamp"]     = timestamp
    global_dict["param_file"]    = param_file
    global_dict["state_transition_file"] = state_transition_file
    return global_dict

def storeFiles(global_dict):
    sfolder = global_dict["source_folder"]
    # dest folder per run
    dfolder = global_dict["dest_folder"] + "results_run%03d/" % global_dict["run"] 
    os.makedirs(dfolder, exist_ok=True)
    shutil.copy(global_dict["param_file"], dfolder)
    shutil.copy("vaccines.csv", sfolder)  # might not exist
    state_transition_file = global_dict["state_transition_file"]
    print("state_tran= ", state_transition_file)
    shutil.copy(global_dict["source_folder"]+"/parameters_0.txt", dfolder)
    shutil.copy(global_dict["base_dest_folder"]+"/data_baseline_p0.txt", dfolder)
    shutil.copy(global_dict["base_dest_folder"]+"/cum_baseline_p0.txt", dfolder)
    shutil.copy(state_transition_file, dfolder)
    #os.remove(global_dict["output_file"])
    os.remove(global_dict["base_dest_folder"]+"/data_baseline_p0.txt")
    os.remove(global_dict["base_dest_folder"]+"/cum_baseline_p0.txt")
    os.remove(state_transition_file)

    with open(dfolder+"global_dict.pkl", "wb") as f:
        pickle.dump(global_dict, f)


def run_simulation(global_dict):
    print("global_dict= ", global_dict)
    dest_folder = global_dict["dest_folder"]
    output_file = global_dict["output_file"]
    #print("output_file= ", output_file); quit()
    os.makedirs(dest_folder, exist_ok=True)
    rpf.readParamFile(global_dict["param_file"], global_dict)

    v1rs = [0, 100, 500, 1000, 5000, 10000, 15000, 20000]
    for run, vac1_rate in enumerate(v1rs):
        print("-----------------------------------------------")
        global_dict["vac1_rate"] = vac1_rate
        global_dict["run"] = run
        args = f"--vac1_rate={vac1_rate}"
        try:
            dfolder = global_dict["dest_folder"] + "results_run%03d/" % run   # dest folder 
            cmd = "./seir %s >& %s" % (args, dfolder + global_dict["output_file"])
            global_dict["cmd"] = cmd
            print("command: ", cmd)
            os.system(cmd)
            storeFiles(global_dict)
        except: 
            print("An error occurred during execution")
            print(f"  run: {run}, vac1_rate: {vac1_rate}\n")
            traceback.print_exc()
            break
        print("global_dict= ", global_dict)
        print("-----------------------------------------------")
        quit()

if __name__ == "__main__":
    global_dict = setupGlobalDict()
    run_simulation(global_dict)
    print(global_dict)
