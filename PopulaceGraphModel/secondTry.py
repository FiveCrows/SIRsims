from rework import *

mask_scalar = 0.3
env_scalars = {"school": 0.3 , "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 16, 'school' :13}
masking = {'workplace': 0, 'school':0, 'household': 0}
distancing = {'workplace': 0, 'school':0, 'household': 0}
preventions = {"masking": masking, "distancing": distancing}
prevention_scalars = {"masking": 0.3}
gamma = 0.1
tau = 0.08

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partition(enumerator, 'age', names)

trans_weighter = TransmissionWeighter(env_scalars, prevention_scalars)
model = PopulaceGraph(trans_weighter, env_degrees, partition, slim = True)
model.build(preventions, env_degrees)
model.simulate(gamma, tau)
print("here")
model.plotSIR()