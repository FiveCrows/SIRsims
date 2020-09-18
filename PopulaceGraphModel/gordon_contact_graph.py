from ModelToolkit import *
import sys
import numpy as np

mask_scalar = 0.3
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 15, 'school': 13}
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

model = PopulaceGraph( partition, slim = False)
model.trans_weighter = trans_weighter

# The simplest approach to symmetrization is Method 1 (M1) in paper by Arregui
def reciprocity(cm, N):
    cmm = np.zeros([4,4])
    for i in range(4):
        for j in range(4):
            if N[i] == 0:
                cmm[i,j] = 0
            else:
                cmm[i,j] = 1/(2 * N[i])  * (cm[i,j]*N[i] + cm[j,i]*N[j])

            if N[j] == 0:
                cmm[j,i] = 0
            else:
                cmm[j,i] = 1/(2 * N[j])  * (cm[j,i]*N[j] + cm[i,j]*N[i])
    return cmm

#---------------------------------------------
def GEclusterBipartite(environment, members_A, members_B, edge_count, weight_scalar = 1, p_random = 0.2):

    edge_list = []  # list of node pairs

    #reorder groups by size
    A = min(members_A, members_B, key = len)
    if A == members_A:
        B = members_B
    else:
        B = members_A

    size_A = len(A)
    size_B = len(B)

    if len(members_A)*len(members_B) < edge_count:
        print("warning, not enough possible edges for cluterBipartite")

    #distance between edge groups
    separation = int(math.ceil(size_B/size_A))

    #size of edge groups and remaining edges
    k = edge_count//size_A
    remainder = edge_count%size_A
    p_random = max(0, p_random - remainder/edge_count)


    for i in range(size_A):
        begin_B_edges = (i * separation - k // 2)%size_B

        for j in range(k):
            if random.random()>p_random:
                B_side = (begin_B_edges +j)%size_B
                edge_list.append((A[i], B[B_side]))
            else:
                edge_list.append((random.choice(A), random.choice(B)))

    for i in range(remainder):
        edge_list.append((random.choice(A), random.choice(B)))

    return edge_list
#---------------------------------------------


def main(n1, n2, m1, m2):
    school_id = 450124041
    cm = model.environments[school_id].contact_matrix[0:4,0:4]
    N = [60, 120, 80, 140]
    cmm = reciprocity(cm, N)
    #print(cm)
    print(cmm)

    lst = lambda n1,n2 : list(range(n1,n2))

    edgeCount = int(cmm[0,1]*N[0])  # at most off by 1

    school_id = 450124041
    #model.clusterBipartite(model.environments[school_id], list(range(0,n1)), list(range(n1,n1+n2)), edgeCount) # fourty four edges will

    edge_list = GEclusterBipartite(cmm, lst(0,n1), lst(n1,n1+n2), edgeCount) 
    print("edge_list\n", edge_list)
    
    # Compute degree
    degree_from = np.zeros(n1, dtype='int')
    degree_to = np.zeros(n2, dtype='int')
    for e in edge_list:
        v1, v2 = e
        degree_from[v1] += 1
        degree_to[v2-n1] += 1

    # Average degree
    print("deg_from= ", degree_from)
    print("deg_to= ", degree_to)
    avg_deg_from = np.mean(degree_from)
    avg_deg_to   = np.mean(degree_to)
    print("max: ", np.max(degree_from))
    print("max: ", np.max(degree_to))
    max_deg = np.max([np.max(degree_from), np.max(degree_to)])
    print("max_deg= ", max_deg)
    
    print("avg_deg_from = ", avg_deg_from)
    print("avg_deg_to = ", avg_deg_to)
    quit()

    # Total degree histogram
    degrees = np.zeros(max_deg+1, dtype='int')  # add 1 for security
    for n in range(n1):
        deg = degree_from[n]
        degrees[deg] += 1
    for n in range(n2):
        deg = degree_to[n]
        degrees[deg] += 1

    print("Degree histogram")
    print("Degree, nb nodes")
    for n in range(max_deg+1):
        print(n+1, degrees[n])


#----------------------------------------
if __name__  == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) == 4:
            argvmain(int(sys.argv[0]), int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("must provide four integers as arguments, n1,n2,m1,m2")
        print("run default: 60, 120, 80, 140")
        main(60, 120, 80, 140)
