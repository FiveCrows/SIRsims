from ge1_modelingToolkit import *
import copy
import os
# OS independence
from shutil import copyfile
# plot chance of infection

# Time stamp to identify the simulation, and the directories where the data is stored
# Called in constructor of Partioning graph (used to be called in simulate() method
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# os has operating system aware functions (provided by Derek)
# https://urldefense.com/v3/__https://www.geeksforgeeks.org/python-os-mkdir-method/__;!!PhOWcWs!nUjA_KItpfwobIwE_tQ_ogPwde2wU4O0EeqeEL0s7bv6kOvIMGkiWbnCzzMVIh3blQ$ 
dstdirname = os.path.join(".","ge_simResults", timestamp, "src")
os.makedirs(dstdirname)

# Independent of OS
copyfile ('ge3_populace_study.py',os.path.join(dstdirname,'ge3_populace_study.py'))
copyfile ('ge1_modelingToolkit.py',os.path.join(dstdirname,'ge1_modelingToolkit.py'))

#################################################################################
#####   Begin setting up model variables  #######################################
#################################################################################

# Run with 10% of the data: slim=True
# Run with all the data: slim=False
slim = True
slim = False

#These values scale the weight that goes onto edges by the environment type involved
# Parameters less than 1 reduce the infectivity
default_env_scalars   = {"school": 1.0, "workplace": 1.0, "household": 1.0}

#As None, the degrees of the environment are implicit in contact matrices
env_degrees           = {'workplace': None, 'school': None}

# Binary variables. Either there are preventions or not

# Preventions refer to the fraction of the population wearing masks, or the fraction of 
# the population practicing social distancing. 

# these numbers refer to the fraction of people with masks and distancing
workplace_preventions = {'masking': 0.0, 'distancing': 0}

#the prevention measures in the schools
school_preventions    = {'masking':0,  'distancing': 0}

#the prevention measures in the schools
household_preventions = {'masking':0,  'distancing': 0}

# Dictionary of dictionaries
#combine all preventions into one var to easily pass during reweight and build
preventions = {'workplace': workplace_preventions, 
               'school': school_preventions, 
               'household': household_preventions}

# Parameters found by Dustin
# these values specify how much of a reduction in weight occurs when people are masked, or distancing
# These parameters are global and not per environment because people carry their own mask
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

#this object holds rules and variables for choosing weights
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)

# https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_SIR.html
# Argument to EoN.fast_SIR(G, tau, gamma, initial_infecteds=None,
gamma  = 0.2  # Recovery rate per edge (EoN.fast_SIR)
tau    = 0.2  # Transmission rate per node (EoN.fast_SIR) (also called beta in the literature)

#the partioner is needed to put the members of each environment into a partition,
#currently, it is setup to match the partition that is implicit to the loaded contact matrices
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)-1) for i in range(15)]
names.append("75-100")
partition = Partitioner('age', enumerator, names)
print("Age brackets: ", names)

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

# Create Graph

#init, build simulate
model = PopulaceGraph(partition, timestamp, slim = slim)
model.build(trans_weighter, preventions, prevention_reductions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')
#----------------------------

# Create a range of simulation
# 'masking': 0.1, 0.2, 0.3, 0.4
# 'distancing': 0.0, 0.1, 0.2, 0.3, 0.4

def reduction_study(s_mask, s_dist, w_mask, w_dist):
    # Probably had no effect since masking and distancing initially set to zero
    # if s_mask = 0, prevention_reduction in school masks won't have an effect, but will in the workforce
    prevent = copy.deepcopy(preventions)
    prevent['school']['masking'] = s_mask
    prevent['school']['distancing'] = s_dist
    prevent['workplace']['masking'] = w_mask
    prevent['workplace']['distancing'] = w_dist
    # value of zero indicate that masking and social distancing have no effect.
    reduce_masking    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];  # np.linspace(0., 1.0, 6)
    reduce_distancing = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    # If s_mask == 0, 
    for m in reduce_masking:
        for d in reduce_distancing:
            #print("m,d= ", m,d)
            prevention_reductions = {'masking': m, 'distancing': d} 
            #print("script: preventions: ", prevent)
            #print("script: prevention_reductions: ", prevention_reductions)
            trans_weighter.setPreventions(prevent)   #### where does Bryan apply self.preventions = preventions? *******
            trans_weighter.setPreventionReductions(prevention_reductions)
            model.reweight(trans_weighter, prevent, prevention_reductions)  # 2nd arg not required because of setPreventions
            model.simulate(gamma, tau, title= "red_mask=%4.2f,red_dist=%4.2f,sm=%2.1f,sd=%2.1fd,wm=%2.1f,wd=%2.1f" % (m, d, s_mask, s_dist, w_mask, w_dist))

s_mask = [0.7, 0.3]  # percentage of people wearing masks in schools
s_dist = [0.7, 0.3]  # percentage of people social distancing in schools
w_mask = [0.7, 0.3]  # percentage of people wearing masks in the workplace
w_dist = [0.7, 0.3]  # percentage of people social distancing in the workplace

# 16 cases * 16 cases for a total of 256 cases
# Note: if s_mask == 0, prevention_reductions won't have an effect
for sm in s_mask:
 for wm in w_mask:
  for sd in s_dist:
   for wd in w_dist:
        reduction_study(sm, sd, wm, wd)
        pass 

quit()


