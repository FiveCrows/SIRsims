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

run = 11    # <<<< Set to create a new run

#dirs = list(os.scandir("graphs"))
#dirs = ["data_ge"]
#print(dirs)

def setupGlobalDict():
    source_folder = "data_ge/"
    base_dest_folder = source_folder + "/results/"  # no final slash
    dest_folder = source_folder + "/run%03d/" % run  # no final slash
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
    global_dict["state_transition_file"] = "transition_stats.csv"
    global_dict["counts_file"]   = "counts.csv"
    return global_dict

def storeFiles(global_dict):
    gd = global_dict
    sfolder = gd["source_folder"]
    print("sfolder= ", sfolder)
    # dest folder per run
    dfolder = gd["dest_folder"] + "results_run%03d/" % gd["run"] 
    print("dfolder= ", dfolder)
    os.makedirs(dfolder, exist_ok=True)
    shutil.copy(gd["param_file"], dfolder)
    # No longer needed
    #shutil.copy("vaccines.csv", sfolder)  # might not exist
    state_transition_file = gd["state_transition_file"]
    print("state_tran= ", state_transition_file)
    shutil.copy(gd["source_folder"]+"/parameters_0.txt", dfolder)
    print("base_dst_folder= ", gd["base_dest_folder"]+"/data_baseline_p0.txt")
    shutil.copy(gd["base_dest_folder"]+"/data_baseline_p0.txt", dfolder)
    shutil.copy(gd["base_dest_folder"]+"/cum_baseline_p0.txt", dfolder)
    shutil.copy(state_transition_file, dfolder)
    shutil.copy(gd["counts_file"], dfolder)
    #os.remove(gd["output_file"])
    os.remove(gd["base_dest_folder"]+"/data_baseline_p0.txt")
    os.remove(gd["base_dest_folder"]+"/cum_baseline_p0.txt")
    os.remove(state_transition_file)
    os.remove(gd["counts_file"])

    src_code_folder = global_dict["dest_folder"] + "/src"
    os.makedirs(src_code_folder, exist_ok=True)
    shutil.copy("G.cpp", src_code_folder)
    shutil.copy("G.h", src_code_folder)
    shutil.copy("head.h", src_code_folder)
    shutil.copy("run_multi_leon_simulations.py", src_code_folder)
    shutil.copy("run_multi_stats.py", src_code_folder)
    shutil.copy("stats.py", src_code_folder)
    shutil.copy("read_data.py", src_code_folder)
    shutil.copy("read_parameter_file.py", src_code_folder)
    shutil.copy("Makefile", src_code_folder)

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
    v1rs = [0, 1000, 10000, 20000]
    for run, vac1_rate in enumerate(v1rs):
        dfolder = global_dict["dest_folder"] + "results_run%03d/" % run
        os.makedirs(dfolder, exist_ok=True)  # necessary
        print("-----------------------------------------------")
        global_dict["run"] = run
        global_dict["vac1_rate"] = vac1_rate
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

if __name__ == "__main__":
    global_dict = setupGlobalDict()
    run_simulation(global_dict)
    print(global_dict)
