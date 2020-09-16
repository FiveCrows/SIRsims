from ModelToolkit import *
import sys

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

def main(n1, n2, m1, m2):
    edgeCountA = n1*m1
    edgeCountB = n2*m2
    edgeCount = round((edgeCountA+edgeCountB)/2)
    if edgeCount>n1*n2:
        print("too many edges to be possible, rounding edge count down")
        edgeCount = n1*n2
    if edgeCountA == edgeCountB:
        print("graph is possible")
    else:
        print("graph not possible, {} edges would be needed to satisfy m1, but {} to satisfy m2, so using {} instead".format(edgeCountA, edgeCountB, edgeCount))

    model.clusterBipartite(model.environments[58758613], list(range(0,n1)), list(range(n1,n1+n2)), edgeCount) # fourty four edges will

    pos = nx.spring_layout(model.graph)
    nx.draw_networkx_nodes(model.graph, pos, nodelist=list(range(0,n1)), node_color='r')
    nx.draw_networkx_nodes(model.graph, pos, nodelist=list(range(n1,n1+n2)), node_color='b')
    nx.draw_networkx_edges(model.graph, pos)
    plt.show()

if __name__  == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) == 4:


            main(int(sys.argv[0]), int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("must provide four integers as arguments, n1,n2,m1,m2")
