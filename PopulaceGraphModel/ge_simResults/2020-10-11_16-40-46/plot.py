import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_pickle("metadata.gz")

row = df.iloc[3]
print(row)

# Plot SIR curves per age as a function of time

sir = row.SIR
S, I, R, t = sir['S'], sir['I'], sir['R'], sir['t']
print(S)
print(t)

#plt.plot(t, S)
#plt.plot(t, I)
#plt.plot(t, R)
#plt.show()


ages = row.ages
print("ages")
print(ages)

k = 3  # bracket 15-19
print(list(ages.keys())); # 0, 10, 20, .., quit()

curves = {}
# I want curves[0-19].S = [S[0], S[10], ...]
# I want curves[0-19].I = [I[0], I[10], ...]

for k in range(0,20):
    curves[k] = {}
    curves[k]['S'] = []
    curves[k]['I'] = []
    curves[k]['R'] = []
    curves[k]['t'] = []

print("---------")
print("curves= ", curves)

for k, v in ages.items():
    ages_l = v
    print(list(v.keys()))  # 1 => 19
    for k_age, v_age in v.items():   # 1 .. 19
       curves[k_age]['S'].append(v_age['S'])
       curves[k_age]['I'].append(v_age['I'])
       curves[k_age]['R'].append(v_age['R'])
       curves[k_age]['t'].append(k)

for k in range(0,20):
    t = curves[k]['t'][0:-1]
    plt.plot(t, curves[k]['I'][0:-1])

plt.show()



plt.plot(t, curves[3]['S'][0:-1])
plt.plot(t, curves[3]['I'][0:-1])
plt.plot(t, curves[3]['R'][0:-1])
plt.show()
print(curves[3]['S'])
