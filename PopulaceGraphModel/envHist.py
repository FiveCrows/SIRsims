#this script is written to test histograms from makeGraph in class Gordon
from genRandEdit import *
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': None, 'school': None}
default_env_masking = {'workplace': 0, 'school':0, 'household': 0}

workplace_preventions = {'masking': 0, 'distancing': 0}
school_preventions = {'masking':0, 'distancing': 0}
household_preventions = {'masking':0, 'distancing':0}
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
prevention_reductions = {'masking': 0.1722, 'distancing': 0.2071}# dustins vals
trans_weighter = TransmissionWeighter(default_env_scalars, prevention_reductions)
gamma = 0.1
tau = 0.08

enumerator = {i:i//5 for i in range(75)}
enumerator.update({i:15 for i in range(75,100)})
names = ["{}:{}".format(5 * i, 5 * (i + 1)) for i in range(15)]
partition = Partitioner(enumerator, 'age', names)

# Choose one or the other model



model = PopulaceGraph( partition, slim = False)
model.environment_degrees = env_degrees
model.trans_weighter = trans_weighter
model.reset()
#schools = list(filter(lambda environment: model.environments[environment].type == 'school' and model.environments[environment].population>25,model.environments))
#workplaces = list(filter(lambda environment: model.environments[environment].type == 'workplace' and model.environments[environment].population>25, model.environments))
schools = sorted(list(filter(lambda environment: model.environments[environment].type == 'school', model.environments)), key = lambda environment: model.environments[environment].population)
workplaces = sorted(list(filter(lambda environment: model.environments[environment].type == 'workplace', model.environments)), key = lambda environment: model.environments[environment].population)

lst = schools
for index in reversed(lst):
    print("pop= ", model.environments[index].population)

# list is a command. Do not use as a variable
for lst in [schools, workplaces]:
    nrows, ncols = 5, 5
    fig, ax = plt.subplots(nrows, ncols)
    num_plots = nrows * ncols
    fig.set_size_inches(18.5, 10.5)
    plot_num = 0

    for index in reversed(lst[-num_plots:]):
        environment = model.environments[index]
        model.reset()
        model.addEnvironment(environment, model.clusterPartitionedRandom)
        people = list(model.graph.adj.keys())
        graph = model.graph
        pop = environment.population
        print("pop= ", pop)
        degreeCounts = [0] * 75

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

