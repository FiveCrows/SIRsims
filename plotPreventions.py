import "./eventDrivenSIR.py"


----------------------------------------------------------------------
# SERIES OF RUNS with associated PLOTS
labels = []
sol = []
[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], full_data = False)

sol.append([t,S,I,R])
labels.append('Uninfected count using Strogatz nets \nwith 50% random edges, control test')

mask_scalar = 0.3
loc_weights = {"school_id": 0.1 , "work_id": 0.2, "sp_hh_id": 1}
weighter = TransmissionWeighter(loc_weights, mask_scalar)
weighter.record(record)

[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.2])
sol.append([t,S,I,R])
labels.append('With 20% random Strogatz nets')

[t, S, I, R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], exemption = 'schools')
sol.append([t,S,I,R])
labels.append('With primary schools closed')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces': 0.5, 'schools': 0.5})
sol.append([t,S,I,R])
labels.append('With 50% public masking')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces':0 , 'schools':1})
sol.append([t,S,I,R])
labels.append('With school masking')

[t, S, I,R] = simulateGraph(clusterStrogatz, EoN.fast_SIR, weighter, [workAvgDegree, 0.5], masking = {'workplaces': 1, 'schools': 0})
sol.append([t,S,I,R])
labels.append('With workplace masking')

#----------------------------------------------
### PLOT RESULTS
nrows = 3
yl = ['S', 'I', 'R']

# Loop through the states S,I,R
for p in range(0,3):
    plt.subplot(nrows,1,p+1)
    # Loop through the simulations
    for i,label in enumerate(labels):
        plt.plot(sol[i][0], sol[i][p+1], label= label)
        plt.ylabel("# %s" % yl[p])
    if p == 1: plt.legend(loc = 1, prop={'size': 7}, framealpha=0.5)
    plt.xlabel("days")
