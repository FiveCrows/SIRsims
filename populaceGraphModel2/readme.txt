This version of the model is different in a few ways:

The entire list of environments, populace, contact matrices, and the partitioner are now all
in one pickle file, which is in the LeonCountyData file. This simplifies modelingToolkit.py,
Will make it easier if we are ever interested in studying different population sets,
and also has additional attributes loaded, such as latitude, longitude, income, and zipcode

defaultParameters are now stored in a script, defaultParameters, which makes it easier to setup
tests, starting with no masking, no distancing, and other default parameters.

more edits are needed for the simulations to run with the vaccination routines and not be bugged.

simpleTest.py is the current script, just to get things working