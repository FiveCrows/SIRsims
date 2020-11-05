# reworked version of vaccination study for the new version of ge_mo*2.py

from ge_modelingToolkit2 import *
import os
import glob


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

gamma = 0.2# The expected time to recovery will be 1/gamma (days)
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

# denotes the fraction of people using masks
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}

glob_dict['enumerator'] = enumerator
glob_dict['prevention_adoptions'] = prevention_adoptions

#initialize populaceGraph
slim = True
slim = False
print("Running with slim= %d" % slim)

glob_dict['slim'] = slim

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 1.0, "workplace": 1.0}
#this dict is used to decide who is masking, and who is distancing
# 0.7 would represent the fraction of the population masking

# Efficacies have the same statistics at the workplace and schools. 
# At home there are no masks or social distancing. Equivalent to efficacy of zero.
# A value of 0 means that masks are not effective at all
# A value of 1 means that a mask wearer can neither infect or be infected. 
prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}  
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

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

model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim, timestamp=timestamp)
model.resetVaccinatedInfected() # reset to default state (for safety)
mask_types = [0.25, 0.5, 0.25]  # Must sum to 1
model.differentiateMasks(mask_types)

glob_dict['mask_types'] = mask_types

# Reduction parameters
# Mask reduction is given by Normal(avg, cv*avg)
#avg_dict={"distancing": 0.4, "mask_eff": 0.4, "contact": 0.0}
# cv is the std / avg. The avg are the prevention_efficacies

# IGNORE "contact" for now, since there is no defined average. 
cv_dict={"distancing": 0.2, "contact": 0.3, "masking": 0.4}
cv_dict = {}
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, cv_dict=cv_dict)

pop_vacc_perc = 0.0
infect_perc = 0.001
glob_dict['pop_vacc_perc'] = pop_vacc_perc
glob_dict['infect_perc'] = infect_perc

model.vaccinatePopulace(perc=pop_vacc_perc)
model.infectPopulace(perc=infect_perc)
model.buildNetworks(netBuilder)
model.simulate(gamma, tau, title = "cv is None")
cv_vals = np.linspace(0., 1., 6) # Coefficient of variation (stdev/mean)
cv_vals = np.linspace(0.1, 1., 5) # Coefficient of variation (stdev/mean)
cv_vals = np.linspace(0., 0.6, 4) # Coefficient of variation (stdev/mean)

glob_dict['comments'] = """
Perform 3 simulations. 
Each simulation vaccinates a top fraction of the schools ranked wrt population.
In every simulation, a different set of people are infected or vaccinated (assuming that vacc_perc > 0)
"""

#------------------------------------------------------------
def oneVaccinationStudy(mask_adopt, dist_adopt, mask_eff, dist_eff):

    prevention_adoptions["school"]      =  \
          {"masking": mask_adopt, "distancing": dist_adopt}
    prevention_adoptions["workplace"]   =  \
          {"masking": mask_adopt, "distancing": dist_adopt} 


    print("enter vaccination study")

    if 1:

       m = mask_eff
       d = dist_eff

       #for nb_wk in [10, 25, 50,100,200,400,600,800]:
       #for nb_wk in [10, 25, 50, 100]:
       #for nb_wk in [1000]:
       for nb_wk in [0, 2, 5, 10, 25, 50, 100, 1000, 5000, 10000, 15000]:
       #for nb_wk in [4000]:
       #for nb_wk in [0]:
        # Make sure that the max nb of schools is at least 5 below the number of 
        # schools, or else there could be crashes when run with slim=True
        #for nb_sch in [0,1,2,3,4, 5, 10, 20, 40, 50]:
        for nb_sch in [0]:
         print("SCRIPT: nb_sch to vaccinate: ",nb_sch)
         #for v_pop_perc in [0., 0.25, 0.5, 0.75, 0.99]
         for v_pop_perc in [0.0]:
          for perc_vacc in [0., 0.25, 0.50, 0.75,  0.99]:
            glob_dict["loop_nb_wk"] = nb_wk
            glob_dict["loop_nb_sch"] = nb_sch
            glob_dict["loop_v_pop_perc"] = v_pop_perc
            glob_dict["loop_perc_vacc"] = perc_vacc
            rho = 0.001
            glob_dict["init_pop_infect"] = rho

            model.resetVaccinatedInfected() # reset to default state (for safety)
            prevention_efficacies = {'masking': m, 'distancing': d} 
            model.infectPopulace(perc=rho)
            perc_vacc_work = perc_vacc
            perc_vacc_school = perc_vacc

            glob_dict['prevention_efficacies']  = prevention_efficacies
            glob_dict['prevention_adoptions'] = prevention_adoptions
            glob_dict["perc_vacc_work"] = perc_vacc_work
            glob_dict["perc_vacc_school"] = perc_vacc_school

            model.setNbTopWorkplacesToVaccinate(nb_wk, perc_vacc_work)  # should set in term of number of available vaccinations
            # make sure first index < nb schools (nb schools, perc vaccinated)vaccinated)
            model.setNbTopSchoolsToVaccinate(nb_sch, perc_vacc_school)  # should set in term of number of available vaccinations
            model.vaccinatePopulace(perc=v_pop_perc)  # random vaccination of populace
            title= "vaccination"
            model.simulate(gamma, tau, title=title, global_dict=glob_dict)
    return 

#----------------------
# 5 levels of vaccination, 5 levels of mask and social distancing. 25 cases.
def oneSetOfVaccinationStudies():
    mask_adopt = 0.5  # all environments
    dist_adopt = 0.5  # all environments
    mask_effic = [0.5, 0.5]  # (avg, std), all environments
    dist_effic = [0.5, 0.5]  # (avg, std), all environments
    
    glob_dict["mask_adopt"] = mask_adopt
    glob_dict["dist_adopt"] = dist_adopt
    glob_dict["mask_effic"] = mask_effic
    glob_dict["dist_effic"] = dist_effic

    if 1:  # replace by loops in other studies
        print("before enter vaccination study")
        oneVaccinationStudy(mask_adopt, dist_adopt, mask_effic, dist_effic)

#----------------------
for sim_rep in range(1):
    glob_dict['loop_sim_rep'] = sim_rep
    print("--------- SIMULATION %d ----------------" % sim_rep)
    oneSetOfVaccinationStudies()

#-------------------------------------------------------------------
quit()
