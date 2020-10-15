import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u

# 2020-10-14
# Take output of analysis .py (a gzipped csv file), and restructure the SIR curves for all the 
# age groups. Compute additional metadata. All for the purposes of future analysis, tabulation, 
# and plotting. 

df = pd.read_pickle("transformed_metadata.gz")
print(df.columns)

print(df.ages)
ages = df.ages
SIR_age = df.SIR_age
SIR = df.SIR 

print("SIR.keys: ", SIR.keys())
print("SIR_age.keys: ", SIR_age.keys())
print("SIR[3].keys: ", SIR[3].keys())
print("SIR_age[3].keys: ", SIR_age[3].keys())
print("SIR_age[3][7].keys: ", SIR_age[3][7].keys())
#print("SIR.t: ", SIR['t'])
#print(SIR_age[3]['t'])
#print(SIR_age[8]['t'])
