Read README_ScriptsShamik.txt to understand usage of different .py files. 

Other files: 

aggregate.py: script that reads the data in the following files: 

    "people_list_serialized.pkl"
    "/households_list_serialized.pkl"
    "/workplaces_list_serialized.pkl"
    "/ContactMatrixWorkplaces.pkl"
    "/schools_list_serialized.pkl"
    "/ContactMatrixSchools.pkl"

and consolidates the data into two files: Leon.pkl, and a slimmed-down version slimmedLeon.pkl. 


Run: The slimmed version eliminates 90% of all people in Leon County

python aggregate.py
-------------------------------------------------------------------------------
