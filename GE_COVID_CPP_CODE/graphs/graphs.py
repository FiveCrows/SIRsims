# Series of different graphs for experimentation
# Each graph will have the same number of nodes, but different constructions. 
# The types considered: 
#  - Configuration model graphs with specified degree distribution
#    A graphs is chosen with uniform probability from all possible graphs
#    - expected_degree_graph() 
#    - configuration_model() 

#  nx.gnm_random_graph(nb_nodes, nb_edges)

from networkx.algorithms import approximation as ap
import matplotlib.pyplot as plt
import networkx as nx
import os, sys

skip_properties = True

def printGraph(G, folder, degree):
    print("printGraph")
    print("nb edges: %d" % G.number_of_edges())
    print("nb nodes: %d" % G.number_of_nodes())

    nb_nodes = G.number_of_nodes()
    nb_edges = G.number_of_edges()
    # Do not include nb edges because it can change for fixed nb nodes and degree
    folder = folder + "_" + str(nb_nodes) + "_" + str(degree)

    try:
        os.mkdir(folder)
    except:
        pass

    os.chdir(folder)

    fd = open("network.txt", "w")
    w = 1.
    fd.write("%d\n" % G.number_of_edges())
    for e in G.edges():
        fd.write("%d %d %f\n" % (e[0], e[1], 1))
    fd.close()
    
    fd = open("nodes.txt", "w")
    fd.write("%d\n" % G.number_of_nodes())
    for n in G.nodes():
        fd.write("%d 1\n" % n)
    fd.close()

    os.chdir("..")


def graphProperties(G, msg, skip=True):
    if skip: return
    print("--------------------------------------------------")
    print("   %s" % msg)
    print("Nodes: ", G.number_of_nodes())
    print("Edges: ", G.number_of_edges())
    print("type(G): ", type(G))
    #mc = ap.max_clique(G)
    #print("computed max_clique: ", mc)
    ac = ap.average_clustering(G)
    print("computed avg_clustering: ", ac)
    bt = nx.betweenness_centrality(G)
    print("bt= ", max(list(bt.values())))
    da = nx.degree_assortativity_coefficient(G)
    print("Degree assortativity: ", da)
    try:
        di = nx.diameter(G)
        print("Diameter: ", di)
    except:
        print("Disconnected graph")
        pass


nb_nodes = 10000
for degree in [2, 5, 10, 20]:
    # The degree is not constant
    nb_edges = degree*nb_nodes

    """
    G = nx.gnm_random_graph(nb_nodes, nb_edges)
    graphProperties(G, "Gnm Random Graph")
    printGraph(G, "gnm_random_graph", degree)

    G = nx.random_regular_graph(degree, nb_nodes)
    graphProperties(G, "Random Regular Graph")
    printGraph(G, "random_regular_graph", degree)

    # The degree distribution is close to what is specified. Not exact. But fast. 
    degree_seq = [degree] * nb_nodes
    G = nx.expected_degree_graph(degree_seq)
    graphProperties(G, "Expected Degree Graph")
    printGraph(G, "expected_degree_graph", degree)
    """

    # Multigraph
    degree_seq = [degree] * nb_nodes
    G = nx.configuration_model(degree_seq)
    G = nx.Graph(G)  # convert Multigraph to graph
    #G.remove_edges_from(nx.selfloop_edges(G))  # remove self-loops
    #print(G.number_of_nodes(), G.number_of_edges())
    graphProperties(G, "Configuration Model")
    printGraph(G, "config_model_graph", degree)


