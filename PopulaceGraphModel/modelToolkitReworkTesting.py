from ge_modelingToolkit2 import *
#from ModelToolkit2 import *

gamma = 0.1# The expected time to recovery will be 1/gamma (days)
tau = 0.2#  The expected time to transmission by an ge will be 1/(weight*Tau) (days)

#construct partitioner
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partitioner = Partitioner('age', enumerator, names)
prevention_prevalences = {"household": {"masking": 0, "distancing": 0},
                          "school": {"masking": 0, "distancing": 0},
                          "workplace": {"masking": 0, "distancing": 0}}

#initialize populaceGraph
slim = False
model = PopulaceGraph(partitioner, prevention_prevalences, slim=slim)
model.differentiateMasks([0.25,0.5,0.25])

#construct netBuilder
#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 0.3, "workplace": 0.3}
#this dict is used to decide who is masking, and who is distancing
prevention_efficacies = {"masking": 0.7, "distancing": 0.7}

netBuilder = NetBuilder(env_type_scalars, prevention_efficacies, {"weight": 0, "contact": 0, "mask_eff": 0})
model.buildNetworks(netBuilder)
model.simulate(gamma, tau, title = "cv is None")
cv_vals = np.arange(0,1,0.1)  # Coefficient of variation (stdev/mean)

for item in netBuilder.cv_dict.items():

    for val in cv_vals:
        netBuilder.cv_dict[item[0]] = val
        model.reweight(netBuilder, prevention_prevalences)
        model.simulate(gamma, tau, title="{} cv = {}".format(item[0], item[1]))
    #return back to normal
    netBuilder.cv_dict[item[0]] = item[1]
    print(model.getPeakPrevalences())
    plt.plot(model.getPeakPrevalences(),label = item[0])

    model.reset()
plt.legend()
plt.show()



