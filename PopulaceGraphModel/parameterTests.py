import ContagionTest
import numpy as np
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :20}
env_masking = {'work': 0, 'school':0, 'household': 0}

gamma = 0.1
tau = 0.08
trans_weighter = TransmissionWeighter(default_env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, slim = True)

#partition ages
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
age_partition,id_to_partition =  model.partition(list(model.graph.nodes),'age', enumerator)

#Test varying env_scalars
scalar_range = np.arange(0, 2, 0.01)
for key in default_env_scalars:
    new_scalars = default_env_scalars
    for scalar in scalar_range:
        new_scalars[key] = scalar
        model.build(model.clusterStrogatz)
        model.simulate(gamma, tau)
        sim = model.sims[-1]
        totals = []
        end_time = sim.t()[-1]
        for element in age_partition:
            totals.append(sum(status == 'S' for status in sim.get_statuses(element, end_time).values()) / len(element))


#idea: plot the bars by age-groups, to represent the portion infected in time
#idea: plot the

#Test varying env_degrees


#Test varying gamma, tau