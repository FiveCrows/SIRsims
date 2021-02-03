# Read degrees_mean_var.csv, created by run_stats.py
# groupby degree, compute means and variances
# The dataframe only contains transitions IS_L

import pandas as pd

L  = 1
IS = 4
R  = 8

df = pd.read_csv("degrees_mean_var.csv")
df1 = df.groupby("degree").mean()
df2 = df.groupby("degree").var()
print("\n\nmean(mean,var): \n", df1)
print("\n\nvar(mean,var): \n", df2)
