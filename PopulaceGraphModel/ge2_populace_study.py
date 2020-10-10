from ge1_modelingToolkit import *
import copy
import os
# plot chance of infection

# Time stamp to identify the simulation, and the directories where the data is stored
# Called in constructor of Partioning graph (used to be called in simulate() method
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dstdirname = "/".join(["ge_simResults", timestamp, "src"])
# Insecure approach
os.system("mkdir -p %s" % dstdirname)
try:
    os.system("cp %s %s" % ('ge2_populace_study.py', dstdirname))
    os.system("cp %s %s" % ('ge1_modelingToolkit.py', dstdirname))
except:
    os.system("copy %s %s" % ('ge2_populace_study.py', dstdirname))
    os.system("copy %s %s" % ('ge1_modelingToolkit.py', dstdirname))

#################################################################################
#####   Begin setting up model variables  #######################################
#################################################################################

# Run with 10% of the data: slim=True
# Run with all the data: slim=False
slim = True
#slim = False

#These values scale the weight that goes onto edges by the environment type involved
default_env_scalars   = {"school": 0.3, "workplace": 0.3, "household": 1}

#As None, the degrees of the environment are implicit in contact matrices
env_degrees           = {'workplace': None, 'school': None}

# Binary variables. Either there are preventions or not

#the prevention measures in the workplaces
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
#these values specify how much of a reduction in weight occurs when people are masked, or distancing
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}

#this object holds rules and variables for choosing weights
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)

# https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_SIR.html
# Argument to EoN.fast_SIR(G, tau, gamma, initial_infecteds=None,
gamma                 = 0.2  # Recovery rate per edge (EoN.fast_SIR)
tau                   = 0.2  # Transmission rate per node (EoN.fast_SIR) (also called beta in the literature)


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
model.build(trans_weighter, preventions, env_degrees)
model.simulate(gamma, tau, title = 'base-test')
#----------------------------

# Create a range of simulation
# 'masking': 0.1, 0.2, 0.3, 0.4
# 'distancing': 0.1, 0.2, 0.3, 0.4
masking    = np.linspace(0.1,0.4,4)
distancing = np.linspace(0.1,0.4,4)
for m in masking:
    for d in distancing:
        prevention_reductions = "{'masking': %f, 'distancing': %f}" % (m, d)
        trans_weighter.setPreventionReductions(prevention_reductions)
        model.simulate(gamma, tau, title= "reductions, mask=%f, distancing=%f" % (m, d))
quit()


#new parameter sets for different tests
school_masks = copy.deepcopy(preventions)
school_masks['school']['masking'] = 1
with_distancing = copy.deepcopy(preventions)
with_distancing['workplace']['distancing'] = 1
#weight with distancing applied to workplace and schools, and resimulate
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'school and workplace distancing')
#----------------------------

#close schools
default_env_scalars['school'] = 0
trans_weighter.setEnvScalars(default_env_scalars)
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed')
#----------------------------

#I change the default preventions here to use but it doesn't matter because this is the last one
preventions['workplace']['masking'] = 1
model.reweight(trans_weighter, with_distancing)
model.simulate(gamma, tau, title = 'schools closed, and workplaces masked')

quit()
pass

#-----------------------------------------
# Simulations ended
#the globalMultiEnvironment is useful if one wants to make the all of leon country
globalMultiEnvironment = model.returnMultiEnvironment(model.environments.keys(), partition)
#some interesting environment to investigate the results in, others in usefulEnvironments.txt
largestWorkplace = model.environments[505001334]
largestWorkplace = model.environments[505001334]
largestSchool = model.environments[450059802]
#bigHousehold = model.environments[58758613]

#this will plot S, I and R curves for the entire populace
model.plotSIR()

#by calling with largestWorkplace, plot will be made for that specific environment
model.plotNodeDegreeHistogram(largestWorkplace)
model.plotContactMatrix(largestWorkplace)

#bug seems to keep coming back in the contactmatrix generator, though I've fixed it, rip, wip...
plt.imshow(largestWorkplace.contact_matrix)

#priority
#Show charts for bipartite n1,n2,m1,m2
#add plots to overleaf
#add description to overleaf

#plot some network-charts
#----------------------
