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
partition = Partitioner('age', enumerator, names)
model = PopulaceGraph( partition, slim = False)
model.trans_weighter = trans_weighter


N = [6000, 12000, 8000, 14000,0,0,0,0,0,0,0,0,0,0,0]

memberReplacement = list(range(sum(N)))
partitionEnumerator = {i: [] for i in range(16)}
peopleAdded = 0

for i in range(len(N)):
    n = N[i]
    set = (list(range(peopleAdded, peopleAdded+ n)))
    partitionEnumerator[i] = set
    peopleAdded = peopleAdded+n

school = model.environments[450124041]
school.members = list(range(sum(N)))
school.partition = partitionEnumerator
school.population = sum(N)
id_to_partition = dict.fromkeys(school.members)
for person in partitionEnumerator[set]:
    id_to_partition[person] = (set)
school.id_to_partition = id_to_partition

print(school.returnReciprocatedCM())
model.environment_degrees = env_degrees
model.addEnvironment(school)
#model.plotNodeDegreeHistogram(syntheticEnvironment)
plt.imshow(school.contact_matrix)
plt.show()
plt.imshow(school.returnReciprocatedCM())
plt.imshow(model.returnContactMatrix())