#this function generates a histogram for household sizes
from ge_modelingToolkit2 import *
import numpy as np


gamma = 0.2# The expected time to recovery will be 1/gamma (days)
tau = 0.2#  The expected time to transmission by an ge will be 1/(weight*Tau) (days)


# Whether or not to save output files  <<<<<<<<<<<<<< Set to save directory
save_output = False
save_output = True
print("save_output= ", save_output)



#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)

# denotes the fraction of people using masks
prevention_adoptions = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}




#initialize populaceGraph
slim = True
slim = False
print("Running with slim= %d" % slim)



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

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

slim = False
model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim)

#initialize empty households bins
h_sizes = [0]*13
s_sizes = [0]*35
w_sizes = [0]*10

for environment in model.environments:
    environment = model.environments[environment]
    if environment.env_type == 'household':
        h_sizes[environment.population]+=1
    if environment.env_type == 'school':
        s_sizes[environment.population//100] +=1
    if environment.env_type == 'workplace':
        w_sizes[int(np.log(environment.population))] +=1

fig,axs = plt.subplots(3)
axs[0].set_title("household size hist")
axs[1].set_title("school size hist")
axs[2].set_title("log-log workplace size hist")
axs[0].bar(range(len(h_sizes)-1),h_sizes[1:])
axs[1].bar(range(len(s_sizes)),s_sizes)
axs[1].set_xticklabels(np.arange(0,100*(40),100))
axs[2].bar(1+np.asarray(range(len(w_sizes))),np.log(w_sizes))
plt.tight_layout()
plt.savefig("plot_house_school_work_stats.pdf")
plt.savefig("plot_house_school_work_stats.jpg")
plt.show()
