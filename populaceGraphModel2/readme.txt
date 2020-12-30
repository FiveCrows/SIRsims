
##################################
Scripting and simulating models
##################################
modelingToolkit.py contains the classes necessary to construct an Environmental-PopulaceGraph model. 
The main class from modelingToolkit is the PopulaceGraph class. 
In order for a PopulaceGraph model to be constructed, it must be able to load pickle files, leon.pkl or slimmedLeon.pkl, within the folder LeonCountyData.
Since these files are large, they are not in the github repository
#############
To generate these pickle files file aggregate.py must be run first
###############
This file has all the contact matrices, people, and environments organized for quick processing. 

In order to quickly load a set of default parameters for the model, the defaultParameters json can be used. 
Once the model is built, its method networkEnvs must be called with a netBuilder object.
Currently, no parameters are  needed to do this. 
This will construct a list of edges for each environment object in the model. 

Then, weightNetwork must be called on the model to pick weights for each edge as it places them into a networkx graph. 
There are three parameters, dicts, which are needed to pick weights for the network
env_type_scalars, prevention_adoptions, and prevention_efficacies
Default values for these are part of the defaultParams json

After the weighted graph is constructed model.infectPopulace, before the model can be simulated, an initial set of infected individuals must be chosen. This is done by calling
infectPopulace on the model, with a float parameter, initial_infected_rate

after all these steps are taken, the model can be simulated with model.simulate(gamma, tau)
where gamma represents the expectation of recovery time, and tau*weight represents the expectation of transmission time for each edge. The model can be rebuilt from any step  with new parameters and resimulated. 

Since some experiments may take a while, particularly if they require repeated simulations, 
results will be compressed and stored in the results folder, within a timestamped directory
(has bug dec 5 2020) 

See simpleTest.py as an example.
###################################
Analyzing Results
###################################
resultAnalysisToolkit is written to contain algorithms for plotting and studying the results of model simulations. However, many of these algorithms are still part of the PopulaceGraph and have yet to be ported. It is a work in progress. 







See
