def saveResults(model, filename, data_dict):
    """
    :param filename: string
    File to save results to
    :param data_dict: dictionary
    Save SIR traces, title, [gamma, tau], preventions
    # save simulation results and metadata to filename
    """

    full_path = "/".join([model.basedir, filename])

    with open(full_path, "wb") as pickle_file:
        pickle.dump(data_dict, pickle_file)

    """
    # reload pickle data
    fd = open(filename, "rb")
    d = pickle.load(fd)
    SIR = d['sim_results']
    print("SIR['t']= ", SIR['t'])
    quit()
    """

        
def aFormatGraph(model, folder):
    ageGroups = [[0,5], [5,18], [18,50], [50,65], [65,100]]
    enumerator = {}
    try:
        os.mkdir(folder)
    except:
        pass
    for i, group in enumerate(ageGroups):
        enumerator.update({j:i for j in range(group[0], group[1])})    
    open(folder +"/nodes.txt","a").writelines(["{} {}\n".format(item,enumerator[model.populace[item]['age']])  for item in model.graph.nodes()])
    i = 0
    with  open(folder +"/edges.txt","a") as file:
        adj = model.graph.adj
        for edgeA in adj:            
            for edgeB in adj[edgeA]:
                file.writelines("{} {} {}\n".format(edgeA,edgeB, adj[edgeA][edgeB]['transmission_weight']))
                i = i+1
    print(i)

class Record:
    def __init__(self, basedir):
        self.log = ""
        self.comments = ""
        self.graph_stats = {}
        self.last_runs_percent_uninfected = 1
        self.basedir = basedir

        try:
            mkdir(self.basedir)
        except:
            pass

    def print(self, string):
        print(string)
        self.log+=('\n')
        self.log+=(string)

    def addComment(self):
        comment = input("Enter comment")
        self.comments += comment
        self.log +=comment

    def printGraphStats(self, graph, statAlgs):
        if not nx.is_connected(graph):
            self.print("graph is not connected. There are {} components".format(nx.number_connected_components(graph)))
            max_subgraph = graph.subgraph(max(nx.connected_components(graph)))
            self.print("{} of nodes lie within the maximal subgraph".format(max_subgraph.number_of_nodes()/graph.number_of_nodes()))
        else:
            max_subgraph = graph
        graphStats = {}
        for statAlg in statAlgs:
            graphStats[statAlg.__name__] = statAlg(max_subgraph)
        self.print(str(graphStats))

    def dump(self):
        #log_txt = open("./ge_simResults/{}/log.txt".format(self.timestamp), "w+")
        log_txt = open(basedir+"/log.txt", "w+")
        log_txt.write(self.log)
        if self.comments != "":
            #comment_txt = open("./ge_simResults/{}/comments.txt".format(self.timestamp),"w+")
            comment_txt = open(basedir+"/comments.txt", "w+")
            comment_txt.write(self.comments)




    #written by Gordon
class Utils:
    def interpolateSIR(self, SIR):
        S = SIR['S']
        I = SIR['I']
        R = SIR['R']
        t = SIR['t']
        print("len(t)= ", len(t))
        # interpolate on daily intervals.
        new_t = np.linspace(0., int(t[-1]), int(t[-1])+1)
        func = interp1d(t, S)
        Snew = func(new_t)
        func = interp1d(t, I)
        Inew = func(new_t)
        func = interp1d(t, R)
        Rnew = func(new_t)
        #print("t= ", new_t)
        #print("S= ", Snew)
        #print("I= ", Inew)
        #print("R= ", Rnew)
        SIR['t'] = new_t
        SIR['S'] = Snew
        SIR['I'] = Inew
        SIR['R'] = Rnew
        return SIR

