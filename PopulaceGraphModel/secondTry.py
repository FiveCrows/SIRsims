from rework import *

mask_scalar = 0.3
env_scalars = {"school": 0.3 , "work": 0.3, "household": 1}
env_degrees = {'work': 16, 'school' :13}
default_env_masking = {'work': 0, 'school':0, 'household': 0}
gamma = 0.1
tau = 0.08

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partition(enumerator, 'age', names)

trans_weighter = TransmissionWeighter(env_scalars, mask_scalar, default_env_masking)
model = PopulaceGraph(trans_weighter, env_degrees, default_env_masking, partition, slim = True)
