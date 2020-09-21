
from ModelToolkit import *
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
which_model = 'random_GE'           # AB contact graph algorithm
#which_model = 'strogatz_AB'         # GE contact graph algorithm (makeGraph)

model = PopulaceGraph( partition, slim = True)

if which_model == 'strogatz_AB':
    model.build(trans_weighter, preventions, env_degrees, alg = model.clusterPartitionedStrogatz)
    name = "Strogatz (Bryan)"
elif which_model == 'random_GE':
    model.build(trans_weighter, preventions, env_degrees, alg = model.clusterPartitionedRandom)
    name = "random edge selection (Gordon)"

#schools = list(filter(lambda environment: model.environments[environment].type == 'school' and model.environments[environment].population>25,model.environments))
#workplaces = list(filter(lambda environment: model.environments[environment].type == 'workplace' and model.environments[environment].population>25, model.environments))
schools = sorted(list(filter(lambda environment: model.environments[environment].type == 'school', model.environments)), key = lambda environment: model.environments[environment].population)
workplaces = sorted(list(filter(lambda environment: model.environments[environment].type == 'workplace', model.environments)), key = lambda environment: model.environments[environment].population)
num_plots = 25




for list in [schools, workplaces]:
    fig, ax = plt.subplots(5, 5)
    fig.set_size_inches(18.5, 10.5)
    plot_num = 0
    for index in reversed(list[-num_plots:]):
        environment = model.environments[index]
        people = environment.members

        graph = model.graph.subgraph(people)

        degreeCounts = [0] * 40
        for person in people:
            try:
                degree = len(graph[person])
            except:
                degree = 0
            degreeCounts[degree] += 1 / environment.population
        ax[plot_num//5, plot_num%5].plot(range(len(degreeCounts)), degreeCounts, label="Population: {}".format(environment.population))
        plot_num +=1

    plt.ylabel("total people")
    plt.xlabel("degree")
    plt.title("histogram for top 25 {}s using ".format(model.environments[list[0]].type) + name)
    plt.legend()
    plt.show()

