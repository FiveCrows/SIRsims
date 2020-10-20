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

"""
3.42605    1.02607    0.017968  0.0579286
 0.513035   5.49482    0.479683  0.0366681
 0.013476   0.719524   2.5271    0.778517
 0.0248265  0.0314298  0.444867  0.274895
"""

cm = syntheticEnvironment.contact_matrix[0:4,0:4]
print(cm)
#plt.imshow(syntheticEnvironment.contact_matrix)
#plt.show()

cm[0,1] = 1.02607
cm[0,2] = 0.017968
cm[0,3] = 0.0579286
cm[1,0] = 0.513035
cm[1,2] = 0.479683
cm[1,3] = 0.0366681
cm[2,0] = 0.013476
cm[2,1] = 0.719524
cm[2,3] = 0.778517
cm[3,0] = 0.0248265
cm[3,1] = 0.0314298
cm[3,2] = 0.444867

# Same contact matrix is the one I am using
print(syntheticEnvironment.contact_matrix[0:4,0:4])

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



