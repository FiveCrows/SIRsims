import glob
import os, sys, shutil
import read_parameter_file as rpf
import pandas as pd
import traceback
import pickle
import itertools

from datetime import datetime

"""
# Global Parameters to adjust at the bottom of this file
    project_nb = 15    # <<<< Set to create a new run
    nb_repeat_runs = 1   # <<<< Set to create a new run
"""

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


def setupGlobalDict(project_nb):
    global_dict = {}
    source_folder = "data_ge/"
    base_dest_folder = source_folder + "/results/"  # no final slash
    dest_folder = source_folder + "/project%05d/" % project_nb  # no final slash 
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
    #shutil.copy("vaccines.csv", sfolder)  # might not exist
    state_transition_file = gd["state_transition_file"]
    print("state_tran= ", state_transition_file)
    shutil.copy(gd["source_folder"]+"/parameters_0.txt", dfolder)
    print("base_dst_folder= ", gd["base_dest_folder"]+"/data_baseline_p0.txt")
    shutil.copy(gd["base_dest_folder"]+"/data_baseline_p0.txt", dfolder)
    shutil.copy(gd["base_dest_folder"]+"/cum_baseline_p0.txt", dfolder)
    print("== list files")
    os.system("ls *.csv")
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
    shutil.copy("timings.py", src_code_folder)
    shutil.copy("Makefile", src_code_folder)
    shutil.copy("breakup_transition.py", gd['dest_folder'])
    print("copy breakup_transition to ", gd['dest_folder'])


#--------------------------------------------------------
def run_simulation(global_dict, project_nb):
    dest_folder = global_dict["dest_folder"]
    output_file = global_dict["output_file"]
    os.makedirs(dest_folder, exist_ok=True)
    rpf.readParamFile(global_dict["param_file"], global_dict)

    ## Make sure the key is in global_dict (so I need a function)
    ## The keys of search_params should be command line arguments
    search_params = global_dict["search_params"]
    out_dicts = dict_product(search_params)

    #----------------------------------------------------------
    nb_repeat_runs = global_dict["nb_repeat_runs"]

    # ATTENTION: Cannot print out_dicts, since it is an iterator
    # This problem might not happen in Julia
    """
    for top_level_run, dct in enumerate(out_dicts):
     for repeat_run in range(nb_repeat_runs):
      print("repeat_run= ", repeat_run)
      print(dct)
      #for top_level_run, dct in enumerate([3,4,5,6,8]): #out_dicts):
      print("top_level_run: %d, repeat_run: %d" % (top_level_run, repeat_run))
    quit()
    """

    # I cannot execute out_dicts anywhere but in an outer list
    for top_level_run, dct in enumerate(out_dicts):
      for repeat_run in range(nb_repeat_runs):
        print("-----------------------------------------------")
        global_dict["top_level_run"] = top_level_run
        run = top_level_run * nb_repeat_runs + repeat_run
        global_dict["run"] = run
        global_dict["leaf_folder"] = "results_run%04d/" % run
        global_dict["repeat_run"] = repeat_run
        dfolder = global_dict["dest_folder"] + global_dict["leaf_folder"]
        os.makedirs(dfolder, exist_ok=True)  # necessary

        print("repeat_run= ", repeat_run)
        print("nb_repeat_runs= ", nb_repeat_runs)
        print("top_level_run= ", top_level_run)
        print("run= ", run)
        print("dct= ", dct)

        args = []
        for k,v in dct.items():
            global_dict[k] = v
            args.append(f"--%s {v}" % k)
        args = " ".join(args)

        # Construct command line
        try:
            # destination folder 
            dfolder = global_dict["dest_folder"] + global_dict["leaf_folder"] 
            cmd = "./seir %s > %s" % (args, dfolder + global_dict["output_file"])
            cmd = "/bin/bash -c  '%s'" % (cmd)
            global_dict["cmd"] = cmd
            print("global_dict: ", global_dict)
            print("==> command: ", cmd)
            os.system(cmd)
            storeFiles(global_dict)
        except: 
            print("An error occurred during execution")
            traceback.print_exc()
            break

        print("--> end loop, repeat_run= ", repeat_run)
        print("--> end loop, nb_repeat_runs= ", nb_repeat_runs)
        print("end loop, top_level_run= ", top_level_run)
        print("dfolder= ", dfolder)

        with open(dfolder+"global_dict.pkl", "wb") as f:
            pickle.dump(global_dict, f)
        print("-----------------------------------------------")

    # preprocess transition files
    # Transition files can be deleted, in principle
    cmd = f"cd {global_dict['dest_folder']}; python breakup_transition.py;"
    print("cmd= ", cmd)
    os.system(cmd)

#----------------------------------------------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

    #===== SET UP PARAMETERS ============
    project_nb = 3    # <<<< Set to create a new run
    nb_repeat_runs = 1   # <<<< Set to create a new run
    search_params = {}
    #search_params['vacc1_rate'] = [0]
    search_params['vacc1_rate'] = [5000, 10000, 20000]
    search_params['vacc1_rate'] = [5000]
    search_params['max_nb_avail_doses'] = [50000]
    search_params['epsilonSinv'] = [2.]
    #search_params['epsilonSinv'] = [0.5, 2., 4.0]
    search_params['muinv'] = [3.0]
    search_params['R0'] = [2., 2.5, 3.0]
    search_params['R0'] = [3.0]
    search_params['beta_shape'] = [2., 5.]
    search_params['beta_shape'] = [2.]
    search_params['beta_scale'] = [3., 5., 7.]
    search_params['beta_scale'] = [3.]
    run_description = f"Project{project_nb}: No vaccinations. Test the shape of the infectivity profile as a function of the Weibull shape and scale parameter values. Test three values of R0." 

    #===== END SET UP PARAMETERS ==================

    #===== NO more CHANGES AFTER THIS POINT =======
    
    global_dict = setupGlobalDict(project_nb)
    global_dict['nb_repeat_runs'] = nb_repeat_runs
    global_dict['project_nb'] = project_nb
    global_dict['search_params'] = search_params
    global_dict['run_description'] = run_description

    run_simulation(global_dict, project_nb)

