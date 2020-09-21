from ModelToolkit import *
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1}
env_degrees = {'workplace': 13, 'school': 13}
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

model = PopulaceGraph( partition, slim = False)
model.build(trans_weighter, preventions, env_degrees)

#schools = list(filter(lambda environment: model.environments[environment].type == 'school' and model.environments[environment].population>25,model.environments))
#workplaces = list(filter(lambda environment: model.environments[environment].type == 'workplace' and model.environments[environment].population>25, model.environments))
schools = sorted(list(filter(lambda environment: model.environments[environment].type == 'school', model.environments)), key = lambda environment: model.environments[environment].population)
workplaces = sorted(list(filter(lambda environment: model.environments[environment].type == 'workplace', model.environments)), key = lambda environment: model.environments[environment].population)

num_plots = 25
for list in [schools, workplaces]:
    for index in list[-1:-num_plots]:
        environment = model.environments[index]
        people = environment.members

        graph = model.graph.subgraph(people)

        degreeCounts = [0] * 25
        for person in people:
            try:
                degree = len(graph[person])
            except:
                degree = 0
            degreeCounts[degree] += 1/environment.population
        plt.plot(range(len(degreeCounts)), degreeCounts, label = "Population: {}".format(environment.population))
    plt.title("histogram for top 25 {}s".format(environment.type))
    plt.ylabel("total people")
    plt.xlabel("degree" )
    plt.legend()
    plt.show()

