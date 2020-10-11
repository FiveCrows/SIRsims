import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as u

df = pd.read_pickle("metadata.gz")

print(df.columns)
row = df.iloc[40]
print(row)

# Plot SIR curves per age as a function of time

sir = row.SIR
S, I, R, t = sir['S'], sir['I'], sir['R'], sir['t']
print("filename: ", row.fn)

S0 = S[0]
plt.plot(t, np.asarray(S)/S[0])
plt.plot(t, np.asarray(I)/S[0])
plt.plot(t, np.array(R)/S[0])
plt.show()

#---------------------
# Choose rows according to some criteria from the pandas table
# For example, wm=1, wd=1, sm=0, sd=0
# and then all the reductions of masks and keeping social reductions constant. 
# or create a 2D plot of max infection as a function of age somehow
#---------------------
curves = u.generateCurves(row)

for k in range(0,20):
    t = curves[k]['t']
    S0 = curves[k]['S'][0]
    if k < 10:
        plt.plot(t, np.asarray(curves[k]['I'])/S0, label=k)
    else:
        plt.plot(t, np.asarray(curves[k]['I'])/S0, linestyle="--", label=k)
plt.legend()

print(df.columns)
print(row)
plt.show()
quit()


plt.plot(t, curves[3]['S'][0:-1])
plt.plot(t, curves[3]['I'][0:-1])
plt.plot(t, curves[3]['R'][0:-1])
plt.show()
print(curves[3]['S'])
