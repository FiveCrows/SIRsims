the classes for building the contagion models are part of ModelToolkit.py

It must be able to load people_list_serialized.pkl, which requires synthdata.py, and it must be able to load files from folder

Contact Matrices, which contains the pickled contact matrices

The most recent script for building and simulating a model, then generating some plots with the toolkit is 

Quick tests can be run by setting slim = True in the model init , otherwise it should be true
PopulaceStudy.py
PopulaceStudy tests three different scenarios currently
The first test has no prevention measures
The second test, schools are closed by setting the school weights to zero
In the third test, the schools are closed and the workplaces are masked

These scenerios can be edited by rechoosing the parameters that go into each model.reweight
I've specifically written populaceStudy to be a good example of the different operations that
ModelToolkit is good fore
if confused on the use of a function, please check docstrings for more info
Many different simulations and plots can easily be built by editing PopulaceStudy.py


Other scripts:
superimposeEnvironmentHistograms plots histograms in a multiplot
 it can also be easily reconfigured to put  all the schools in one plot, and all the workplaces in another.

testContactMatrices.py was written to make sure the contact matrix from an environments actual weights
matches the contact matrix passed as an argument closely

bipartiteInvestigation.py was written to make sure the bipartite algorithm was working effectively.
It needs to be called with four integers, the n1,n2,m1,m2, which is the number of people and average degrees for bipartite members


