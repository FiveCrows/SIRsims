# Author: Bryan Azbill
# Modifications: Gordon Erlebacher

#this function generates a histogram for household sizes
from ge_modelingToolkit2 import *
import numpy as np
import matplotlib

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
slim = False
print("Running with slim= %d" % slim)

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
#doesn't really matter, just to put in place
env_type_scalars = {"household": 1, "school": 1.0, "workplace": 1.0}
prevention_efficacies = {"masking": [0.3,0.3], "distancing": [0.9,0.9]}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

#################################################################################
#####   End setting up model variables  #######################################
#################################################################################

def plotContactMatrix(self, partitioner, env_indices, title = "untitled", figure = None):
        '''
        This function plots the contact matrix for a structured environment
        :param p_env: must be a structured environment
        '''
        global model

        contact_matrix = model.getContactMatrix(partitioner,env_indices)
        plt.imshow(contact_matrix)
        plt.title("Contact Matrix for {}".format(title))
        labels = partitioner.labels
        if labels == None:
            labels = ["{}-{}".format(5 * i, (5 * (i + 1))-1) for i in range(15)]
        axisticks= list(range(15))
        plt.xticks(axisticks, labels, rotation= 'vertical')
        plt.yticks(axisticks, labels)
        plt.xlabel('Age Group')
        plt.ylabel('Age Group')

def getContactMatrix(self, partitioner, env_indices):
        n_sets = partitioner.num_sets
        cm = np.zeros([n_sets, n_sets])
        setSizes = np.zeros(n_sets)
        #add every
        for index in env_indices:
            env = self.environments[index]
            #assigns each person to a set
            placements, partition = partitioner.placeAndPartition(env.members, self.populace)
            setSizes += np.array([len(partition[index]) for index in partition])
            for edge in env.edges:
                cm[placements[edge[0]], placements[edge[1]]] += 1
        cm = np.nan_to_num([np.array(row)/setSizes for row in cm])
        return cm


slim = False
netBuilder = NetBuilder(env_type_scalars, prevention_efficacies)
model = PopulaceGraph(partitioner, prevention_adoptions, prevention_efficacies, slim=slim)
model.buildNetworks(netBuilder)
households = model.listEnvByType('household')
schools = model.listEnvByType('school')
workplaces = model.listEnvByType('workplace')

matplotlib.rc('font', size=16)
matplotlib.rc('xtick', labelsize=12) # axis tick labels
matplotlib.rc('ytick', labelsize=12) # axis tick labels
matplotlib.rc('axes', labelsize=12)  # axis label
matplotlib.rc('axes', titlesize=12)  # subplot title
matplotlib.rc('figure', titlesize=12)

cm_home = model.plotContactMatrix(partitioner, households, "All households", createPlot=False)
cm_school = model.plotContactMatrix(partitioner, schools, "All schools", createPlot=False)
cm_work = model.plotContactMatrix(partitioner, workplaces, "All workplaces", createPlot=False)

# PLOTS *****
fig, axes = plt.subplots(2,2)
axes = axes.reshape(-1)
plt.suptitle("Contact Matrices", fontsize=16)

ax = axes[0]
ax.imshow(cm_school)
ax.set_xlabel("age bracket")
ax.set_ylabel("age bracket")
ax.set_title("All Schools")

ax = axes[1]
ax.imshow(cm_work)
ax.set_xlabel("age bracket")
ax.set_ylabel("age bracket")
ax.set_title("All workplaces")

ax = axes[2]
ax.imshow(cm_home)
ax.set_xlabel("age bracket")
ax.set_ylabel("age bracket")
ax.set_title("All households")

axes[-1].axis('off')

plt.tight_layout()
plt.show()

