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
save_output = True
save_output = False
print("save_output= ", save_output)

glob_dict['save_output'] = save_output

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 1, "distancing": 1},
                          "workplace": {"masking": 1, "distancing": 1}}

glob_dict['enumerator'] = enumerator
glob_dict['prevention_adoptions'] = prevention_adoptions

#initialize populaceGraph
slim = False
slim = True
print("Running with slim= %d" % slim)

glob_dict['slim'] = slim

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 1.0, "workplace": 1.0}
#this dict is used to decide who is masking, and who is distancing
# 0.7 would represent the fraction of the population masking
# A value of 0 means that nobody is wearing masks
prevention_efficacies = {"masking": 0.7, "distancing": 0.7}

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

model = PopulaceGraph(partitioner, prevention_adoptions, slim=slim, timestamp=timestamp)
model.resetVaccinated_Infected() # reset to default state (for safety)
mask_types = [0.25, 0.5, 0.25]  # Must sum to 1
model.differentiateMasks(mask_types)

glob_dict['mask_types'] = mask_types

# Reduction parameters
# Mask reduction is given by Normal(avg, cv*avg)
avg_dict={"distancing": 0.4, "mask_eff": 0.4, "contact": 0.0}
cv_dict={"distancing": 0.2, "contact": 0.3, "mask_eff": 0.4}
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, cv_dict=cv_dict)

vacc_perc = 0.0
infect_perc = 0.001
glob_dict['vacc_perc'] = vacc_perc
glob_dict['infect_perc'] = infect_perc

model.vaccinatePopulace(perc=vacc_perc)
model.infectPopulace(perc=infect_perc)
model.buildNetworks(netBuilder)
model.simulate(gamma, tau, title = "cv is None")
cv_vals = np.linspace(0., 1., 6) # Coefficient of variation (stdev/mean)
cv_vals = np.linspace(0.1, 1., 5) # Coefficient of variation (stdev/mean)

glob_dict['comments'] = """
Perform 5 simulations. 
Each simulation considers 10 values of the coefficient of variation (cv) [0-1]. 
There are no vaccinations. 
The simulations consider separately the effect of masks, contact matrices, and weights (social distancing). 
The graph structure is the same across all simulations, which is not quite correct when the contact 
  matrices are varied. 
In every simulation, a different set of people are infected or vaccinated (assuming that vacc_perc > 0)
What are prevention_efficacies? 
What are prevention_adoptions? 
What are env_type_scalars?
How are masked types used in the code? 
"""

def runGauntlet(count):
  glob_dict['simulation_index'] = count
  # All global variables are accessible
  for cv_key, cv_val in netBuilder.cv_dict.items():
    for val in cv_vals:
        if cv_key != 'mask_eff': continue    ### FOR TESTING
        model.resetVaccinated_Infected() # reset to default state (for safety)
        netBuilder.cv_dict[cv_key] = val
        netBuilder.setModel(model)  # must occur before reweight
        pe = prevention_efficacies
        # Following two lines must occur before reweight
        # a,b of beta distribution hardcoded to 10,10
        model.setupMaskReduction(pe['masking'], cv_dict['mask_eff'])
        model.setupSocialDistanceReduction(pe['distancing'], cv_dict['distancing'])
        # netBuilder has access to model, and model has access to netBuilder. Dangerous.
        model.reweight(netBuilder, prevention_adoptions)
        glob_dict['vacc_perc'] = vacc_perc
        glob_dict['infect_perc'] = infect_perc
        glob_dict['cv_key'] = cv_key
        glob_dict['cv_val'] = val
        avg_key = cv_key
        avg_val = avg_dict[avg_key]
        glob_dict['avg_key'] = avg_key
        glob_dict['avg_val'] = avg_val
        model.infectPopulace(perc=infect_perc)
        model.vaccinatePopulace(perc=vacc_perc)
        model.simulate(gamma, tau, title="{} cv = {}".format(cv_key, cv_val), global_dict=glob_dict)

    netBuilder.cv_dict[cv_key] = cv_val
    print(model.getPeakPrevalences())
    #plt.plot(model.getPeakPrevalences(),label = item[0])


for count in range(5):
    runGauntlet(count)


plt.legend()
plt.show()

