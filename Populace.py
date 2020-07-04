import networkx as nx
class Populace:
    def __init__(self, populace, home_infectivity, school_infectivity, work_infectivity):
        self.graph = nx.graph()
        self.home_infectivity = home_infectivity
        self.school_infectivity = school_infectivity
        self.work_infectivity = work_infectivity

    # connect list of groups with weight
    # TODO update to use a weight calculating function

    def clusterDense(graph, group, memberCount, weight, params):
        # memberWeightScalar = np.sqrt(memberCount)
        for i in range(memberCount):
            for j in range(i):
                graph.add_edge(group[i], group[j], transmission_weight=weight)  # / memberWeightScalar)

    def clusterRandom(self, group, memberCount, weight, params):
        avg_degree = params
        if avg_degree >= memberCount:
            clusterDense(graph, group, memberCount, weight, params)
            return
        edgeProb = 2 * avg_degree / (memberCount - 1)
        subGraph = nx.fast_gnp_random_graph(memberCount, edgeProb)
        relabel = dict(zip(range(memberCount), group))
        nx.relabel.relabel_nodes(subGraph, relabel)
        graph.add_edges_from(subGraph.edges(), transmission_weight=weight)



    def clusterDegree_p(graph, group, memberCount, weight, params):
        degree_p = params
        connectorList = []
        for i in range(memberCount):
            nodeDegree = random.choices(range(len(degree_p)), weights=degree_p)
            connectorList.extend([i] * nodeDegree[0])
        random.shuffle(connectorList)
        # this method DOES leave the chance adding duplicate edges
        i = 0
        while i < len(connectorList) - 1:
            graph.add_edge(group[connectorList[i]], group[connectorList[i + 1]],
                           transmission_weight=weight)
            i = i + 2


    # def clusterPartition(graph, group, memberCount, weight, params):
    def clusterStrogatz(self, group, memberCount, weight, params):
        group.sort()
        local_k = params[0]
        rewire_p = params[1]
        if (local_k % 2 != 0):
            record.print("Error: local_k must be even")
        if local_k >= memberCount:
            clusterDense(graph, group, memberCount, weight, params)
            return

        for i in range(memberCount):
            nodeA = group[i]
            for j in range(1, local_k // 2 + 1):
                if j == 0:
                    continue
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = group[(i + random.choice(range(memberCount - 1))) % memberCount]

                else:
                    nodeB = group[(i + j) % memberCount]
                graph.add_edge(nodeA, nodeB, transmission_weight=weight)


    def clusterByDegree_p(self, groups, weight, degree_p):
        # some random edges may be duplicates, best for large groups
        connectorList = []

        for key in groups.keys():
            if key != None:
                memberCount = len(groups[key])
                connectorList = []
                for i in range(memberCount):
                    nodeDegree = random.choices(range(len(degree_p)), weights=degree_p)
                    connectorList.extend([i] * nodeDegree[0])
                random.shuffle(connectorList)

                i = 0
                while i < len(connectorList) - 1:
                    graph.add_edge(groups[key][connectorList[i]], groups[key][connectorList[i + 1]],
                                   transmission_weight=weight)
                    i = i + 2


    def clusterWith_gnp_random(self, classifier, weight, avg_degree):
        groups = popsByCategory[classifier]
        initial_weights = graph.size()
        for key in groups.keys():
            if key != None:
                memberCount = len(groups[key])
                if (memberCount <= avg_degree):
                    clusterDenseGroup(graph, {0: groups[key]}, weight)
                    continue
                edgeProb = (memberCount * avg_degree) / (memberCount * (memberCount - 1))
                subGraph = nx.fast_gnp_random_graph(memberCount, edgeProb)
                graph.add_edges_from(subGraph.edges(), transmission_weight=weight)

        final_weights = graph.size()
        weights_added = initial_weights - final_weights
        record.printAndRecord(
            "{} weights of size {} have been added for {} work environments".format(weights_added, weight, len(
                popsByCategory[classifier].keys())))

