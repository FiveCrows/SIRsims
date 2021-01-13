import glob
import os, sys, shutil
import read_parameter_file as rpf
import pandas as pd
import traceback
import pickle
import itertools

from datetime import datetime
timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

folders = []
global_dict = {}

run = 12    # <<<< Set to create a new run

#----------------------------------------------------------
# Code posted at
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
# Input: a list of dictionaries
# Output: a list of all possible dicionaries
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
#----------------------------------------------------------


search_params = {}
search_params['vacc1_rate'] = [1000, 5000, 10000]
search_params['max_nb_avail_doses'] = [10000, 50000, 100000]
search_params['epsilonSinv'] = [0.5, 3.]


def setupGlobalDict(run):
    source_folder = "data_ge/"
    base_dest_folder = source_folder + "/results/"  # no final slash
    dest_folder = source_folder + "/run%05d/" % run  # no final slash 
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
    dfolder = gd["dest_folder"] + "results_run%04d/" % gd["run"] 
    print("dfolder= ", dfolder)
    os.makedirs(dfolder, exist_ok=True)
    shutil.copy(gd["param_file"], dfolder)
    shutil.copy("vaccines.csv", sfolder)  # might not exist
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
    shutil.copy("run_multi_leon_sims_1.py", src_code_folder)
    shutil.copy("run_multi_stats.py", src_code_folder)
    shutil.copy("stats.py", src_code_folder)
    shutil.copy("read_data.py", src_code_folder)
    shutil.copy("read_parameter_file.py", src_code_folder)
    shutil.copy("Makefile", src_code_folder)

    with open(dfolder+"global_dict.pkl", "wb") as f:
        pickle.dump(global_dict, f)

#--------------------------------------------------------
def run_simulation(global_dict, run):
    dest_folder = global_dict["dest_folder"]
    output_file = global_dict["output_file"]
    os.makedirs(dest_folder, exist_ok=True)
    rpf.readParamFile(global_dict["param_file"], global_dict)

    ## Make sure the key is in global_dict (so I need a function)
    ## The keys of search_params should be command line arguments
    search_params = {}
    search_params['vacc1_rate'] = [1000, 5000, 10000]
    search_params['max_nb_avail_doses'] = [10000, 50000, 100000]
    search_params['epsilonSinv'] = [0.5, 3.]
    out_dicts = dict_product(search_params)

    for run, dct in enumerate(out_dicts):
        global_dict["leaf_folder"] = "results_run%04d/" % run
        dfolder = global_dict["dest_folder"] + global_dict["leaf_folder"]
        os.makedirs(dfolder, exist_ok=True)  # necessary
        print("-----------------------------------------------")
        global_dict["run"] = run
        args = []
        for k,v in dct.items():
            global_dict[k] = v
            args.append(f"--%s {v}" % k)
        args = " ".join(args)

        # Construct command line
        try:
            # destination folder 
            dfolder = global_dict["dest_folder"] + global_dict["leaf_folder"] 
            cmd = "./seir %s >& %s" % (args, dfolder + global_dict["output_file"])
            global_dict["cmd"] = cmd
            print("command: ", cmd)
            os.system(cmd)
            storeFiles(global_dict)
        except: 
            print("An error occurred during execution")
            traceback.print_exc()
            break
        print("global_dict= ", global_dict)
        print("-----------------------------------------------")

#----------------------------------------------------------------
if __name__ == "__main__":
    global_dict = setupGlobalDict(run)
    run_simulation(global_dict, run)
