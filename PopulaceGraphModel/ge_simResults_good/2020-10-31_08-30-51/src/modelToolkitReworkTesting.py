from ge_modelingToolkit2 import *
import os
import glob
#from ModelToolkit2 import *

# os has operating system aware functions (provided by Derek)
# https://urldefense.com/v3/__https://www.geeksforgeeks.org/python-os-mkdir-method/__;!!PhOWcWs!nUjA_KItpfwobIwE_tQ_ogPwde2wU4O0EeqeEL0s7bv6kOvIMGkiWbnCzzMVIh3blQ$ 

# Time stamp to identify the simulation, and the directories where the data is stored
# Called in constructor of Partioning graph (used to be called in simulate() method
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#################################################################################
#####   Begin setting up model variables  #######################################
#################################################################################

# Global dictionary to store parameters (different for each study)
glob_dict = {}

gamma = 0.1# The expected time to recovery will be 1/gamma (days)
tau = 0.2#  The expected time to transmission by an ge will be 1/(weight*Tau) (days)

glob_dict['gamma'] = gamma
glob_dict['tau'] = tau

# Whether or not to save output files  <<<<<<<<<<<<<< Set to save directory
save_output = False
save_output = True
print("save_output= ", save_output)

glob_dict['save_output'] = save_output

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)
prevention_prevalences = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}

glob_dict['enumerator'] = enumerator
glob_dict['prevention_prevalences'] = prevention_prevalences

#initialize populaceGraph
slim = False
slim = True
print("Running with slim= %d" % slim)

glob_dict['slim'] = slim

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 0.3, "workplace": 0.3}
#this dict is used to decide who is masking, and who is distancing
prevention_efficacies = {"masking": 0.2, "distancing": 0.2}

glob_dict['env_type_scalars'] = env_type_scalars
glob_dict['prevention_efficacies'] = prevention_efficacies

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

if save_output:
    dstdirname = os.path.join(".", "ge_simResults", timestamp, "src")
    os.makedirs(dstdirname)
    # Independent of OS
    os.system("cp *.py %s" % dstdirname)  # not OS independent
    #copyfile ('ge1_modelToolkit.py',os.path.join(dstdirname,'ge1_modelToolkit.py'))

model = PopulaceGraph(partitioner, prevention_prevalences, slim=slim, timestamp=timestamp)
model.resetVaccinated_Infected() # reset to default state (for safety)
mask_types = [0.25, 0.5, 0.25]  # Must sum to 1
model.differentiateMasks(mask_types)

glob_dict['mask_types'] = mask_types

netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, cv_dict={"weight": 0, "contact": 0, "mask_eff": 0})

vacc_perc = 0.0
infect_perc = 0.001
glob_dict['vacc_perc'] = vacc_perc
glob_dict['infect_perc'] = infect_perc

model.vaccinatePopulace(perc=vacc_perc)
model.infectPopulace(perc=infect_perc)
model.buildNetworks(netBuilder)
model.simulate(gamma, tau, title = "cv is None")
cv_vals = np.linspace(0., 1., 6) # Coefficient of variation (stdev/mean)

for item in netBuilder.cv_dict.items():
    for val in cv_vals:
        model.resetVaccinated_Infected() # reset to default state (for safety)
        netBuilder.cv_dict[item[0]] = val
        model.reweight(netBuilder, prevention_prevalences)
        glob_dict['vacc_perc'] = vacc_perc
        glob_dict['infect_perc'] = infect_perc
        glob_dict['cv_key'] = item[0]
        glob_dict['cv_val'] = val
        model.infectPopulace(perc=infect_perc)
        model.vaccinatePopulace(perc=vacc_perc)
        model.simulate(gamma, tau, title="{} cv = {}".format(item[0], item[1]), global_dict=glob_dict)

    #return back to normal
    netBuilder.cv_dict[item[0]] = item[1]
    print(model.getPeakPrevalences())
    #plt.plot(model.getPeakPrevalences(),label = item[0])

plt.legend()
plt.show()

