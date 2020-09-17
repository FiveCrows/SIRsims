from ModelToolkit import *
# plot chance of infection
mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 15, 'school': 20}
default_env_masking = {'workplace': 0, 'school':0, 'household': 0}
workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions = {'masking':0, 'distancing': 0}
household_preventions = {'masking':0, 'distancing':0}
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins values
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
gamma = 0.1
tau = 0.08
enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner(enumerator, 'age', names)
model = PopulaceGraph( partition, slim = True)
model.trans_weighter = trans_weighter


N = [600, 1200, 800, 1400,0,0,0,0,0,0,0,0,0,0,0]
syntheticEnvironment = model.environments[450124041]
plt.imshow(syntheticEnvironment.contact_matrix)
plt.show()
memberReplacement = list(range(sum(N)))
partitionEnumerator = {i: [] for i in range(16)}
peopleAdded = 0

for i in range(len(N)):
    n = N[i]
    set = (list(range(peopleAdded, peopleAdded+ n)))
    partitionEnumerator[i] = set
    peopleAdded = peopleAdded+n

syntheticEnvironment.members = list(range(sum(N)))
syntheticEnvironment.partitioned_members = partitionEnumerator
syntheticEnvironment.population = sum(N)
model.environment_degrees = env_degrees

model.addEnvironment(syntheticEnvironment)
model.plotNodeDegreeHistogram(syntheticEnvironment)



