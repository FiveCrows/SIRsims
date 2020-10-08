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
Check docstrings for more info
Many different simulations and plots can easily be built by editing PopulaceStudy.py


Other scripts:
superimposeEnvironmentHistograms superimposes histograms for all the schools in one plot, and all the workplaces in another.
It also has code, in comments, which produces