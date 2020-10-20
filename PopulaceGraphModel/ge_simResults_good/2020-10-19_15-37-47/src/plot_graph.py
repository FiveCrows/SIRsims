import matplotlib.pyplot as plt

def plotGraph(model, schools, workplaces):
    # list is a command. Do not use as a variable
    for lst in [schools, workplaces]:
        nrows, ncols = 5, 5
        fig, ax = plt.subplots(nrows, ncols)
        num_plots = nrows * ncols
        fig.set_size_inches(18.5, 10.5)
        plot_num = 0
    
        for index in reversed(lst[-num_plots:]):
            environment = model.environments[index]
            people = environment.members
            pop = environment.population
            print("pop= ", pop)
    
            graph = model.graph.subgraph(people)
            degreeCounts = [0] * 40
    
            for person in people:
                try:
                    degree = len(graph[person])
                except:
                    degree = 0
                degreeCounts[degree] += 1 / pop
    
            axx = ax[plot_num//5, plot_num%5]
            axx.plot(range(len(degreeCounts)), degreeCounts,
                 label="Population: {}".format(pop))
            axx.set_title("(GE) N="+str(pop))
            plot_num += 1
    
        plt.ylabel("total people")
        plt.xlabel("degree")
        #plt.title("histogram for top 25 {}s using ".format(model.environments[lst[0]].type) + name)
        plt.legend()
        plt.show()
