import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u

# 2020-10-14
# Take output of analysis .py (a gzipped csv file), and restructure the SIR curves for all the 
# age groups. Compute additional metadata. All for the purposes of future analysis, tabulation, 
# and plotting. 

df = pd.read_pickle("metadata.gz")


# https://www.instapaper.com/read/1351126800
# Efficiency of loops
nb_age_groups = 19
rows = {}
N_col = []
curves_col = []
maxI_col = []
#print(df.columns)

for row_tuple in df.itertuples(): 
    N = []
    maxI = []
    curves = u.generateCurves(row_tuple)
    for k in range(nb_age_groups):
        maxI.append( np.max(curves[k]['I']) )
        N.append( curves[k]['S'][0] + curves[k]['I'][0] + curves[k]['R'][0] )
    N_col.append(N)
    curves_col.append(curves)
    maxI_col.append(maxI)
    # add columns for N and curves: that is another 38 columns
    # remove column

dfc = df.copy()
#print("columns: ", dfc.columns)
dfc.drop("ages", axis=1)
#print("columns: ", dfc.columns)

dfc['N_age'] = N_col
dfc['maxI_age'] = maxI_col
dfc['SIR_age'] = curves_col


#print(dfc.head(10))
#print("columns: ", dfc.columns)
print(dfc.maxI_age)
print(dfc.N_age)
dfc.to_pickle("transformed_metadata.gz")
dfc.to_csv("transformed_metadata.csv")

print("dfc.columns: ", dfc.columns)
quit()

