from ModelToolkit import *
import sys
import numpy as np
import random

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

#-----------------------------------------
def reciprocity(cm, N):
    # The simplest approach to symmetrization is Method 1 (M1) in paper by Arregui
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
# This is GE's algorithm, a copy of what I implemented in Julia. 
# We need to try both approaches for our paper. 
def makeGraph(N, index_range, cmm):
    # N: array of age category sizes
    # index_range: lo:hi tuple 
    # cmm: contact matrix with the property: cmm[i,j]*N[i] = cmm[j,i]*N[j]
    # Output: a list of edges to feed into a graph

    edge_list = []
    Nv = sum(N)
    if Nv < 25: return edge_list # <<<<< All to all connection below Nv = 25. Not done yet.

    lo, hi = index_range
    # Assign age groups to the nodes. Randomness not important
    # These are also the node numbers for each category, sorted
    age_bins = [np.repeat([i], N[i]) for i in range(lo,hi)]

    # Efficiently store cummulative sums for age brackets
    cum_N = np.append([0], np.cumsum(N))

    ddict = {}
    total_edges = 0

    print("lo,hi= ", lo, hi)
    for i in range(lo,hi):
        for j in range(lo,i+1):
            #print("lo,i= ", lo, i)
            ddict = {}
            Nij = int(N[i] * cmm[i,j])
            print("i,j= ", i, j, ",    Nij= ", Nij)

            if Nij == 0:
                continue 

            total_edges += Nij
            # List of nodes in both graphs for age brackets i and j
            Vi = list(range(cum_N[i], cum_N[i+1]))  # Check limits
            Vj = list(range(cum_N[j], cum_N[j+1]))  # Check limits

            # Treat the case when the number of edges dictated by the
            # contact matrices is greater than the number of available edges
            # The connectivity is then cmoplete
            lg = len(Vi)
            nbe = lg*(lg-1) // 2

            if Vi == Vj and Nij > nbe:
                Nij = nbe

            count = 0

            while True:
                # p ~ Vi, q ~ Vj
                # no self-edges
                # only reallocate when necessary (that would provide speedup)
                # allocate 1000 at t time
                #p = getRand(Vi, 1) # I could use memoization
                #q = getRand(Vi, 1) # I could use memoization

                #p = rand(Vi, 1)[]
                #q = rand(Vj, 1)[]
                p = random.choice(Vi)
                q = random.choice(Vj)

                if p == q: continue 

                # multiple edges between p,q not allowed
                # Dictionaries only store an edge once
                if p <  q:
                    ddict[(p,q)] = 1
                else:
                    ddict[(q,p)] = 1

                # stop when desired number of edges is reached
                lg = len(ddict)
                if lg == Nij: break 

            for k in ddict.keys():
                s, d = k
                edge_list.append((s,d))

    print("total_edges: ", total_edges)
    print("size of edge_list: ", len(edge_list))
    return edge_list

#------------------------------------------------------------------
# This is Bryan Azbill's routine slightly modified for GE
def GEclusterBipartite(environment, members_A, members_B, edge_count, weight_scalar = 1, p_random = 0.2):

    edge_list = []  # list of node pairs

    #reorder groups by size

    if len(members_A) < len(members_B):
        A, B = members_A, members_B
    else:
        B, A = members_A, members_B

    size_A = len(A)
    size_B = len(B)

    if size_A * size_B < edge_count:
        print("warning, not enough possible edges for cluterBipartite")

    #distance between edge groups
    separation = int(math.ceil(size_B/size_A))

    #size of edge groups and remaining edges
    # I have no idea what this means!!! GE
    k = edge_count // size_A
    remainder = edge_count % size_A
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
    N = [600, 1200, 800, 1400]
    cmm = reciprocity(cm, N)

    lst = lambda n1,n2 : list(range(n1,n2))

    edgeCount = int(cmm[0,1]*N[0])  # at most off by 1

    school_id = 450124041
    #model.clusterBipartite(model.environments[school_id], list(range(0,n1)), list(range(n1,n1+n2)), edgeCount) # fourty four edges will

    # Translated from GE Julia code makeGraph
    edge_list = makeGraph(N, (0,4), cmm);
    print("Total number of edges: ", len(edge_list))

    # node degrees 
    degrees = np.zeros(sum(N), dtype='int')
    degree_hist = np.zeros(100, dtype='int')
    for i,j in edge_list:
        degrees[i] += 1
        degrees[j] += 1

    # degree histogram
    for d in degrees:
        degree_hist[d] += 1

    print("\nDegree, #nodes of that degree")
    for i in range(25):
        print(i, degree_hist[i])

    # Reconstruct contact matrix

    quit()


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
    exit()

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
