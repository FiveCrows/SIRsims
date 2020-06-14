using LightGraphs
using SimpleWeightedGraphs

# Use a const immutable named tuple for efficiency
# I can change values within this constant, but I cannot add another member
# When developing the code, use non-consts in the global space
p = (
    epidemic_sims = 5,
    households = 5,
    household_size = 4,
    class_size = 20,
    workgroup_size = 10,
    employment_rate = 0.9,
    recovery_rRate = 1,
    global_infection_rate = 1,
    house_infectivity = .1,
    work_infectivity = .05,
)
population = p.household_size * p.households
work_classes = [:default, :unemployed, :school,]

age_groups = Int[]  # USED BY PROGRAM?
tau = 1 #transmission factor
gamma = 1.
initial_infected = 1
G = SimpleGraph()

import Random
import Distributions
categ = Distributions.Categorical
## generate population
citizens = 1:population
house_numbers = [i%p.households for i in citizens]

age_choices = [
   (1,5),
   (6,18),
   (19,65),
   (66,120)]
age_weights = [0.05, 0.119, 0.731, 0.1]  # from census.gov for tallahassee 2019
categ = Distributions.Categorical
age_indexes = rand(categ(age_weights),population)  # in [1,length(ageChoices)
age_brackets = age_choices[age_indexes]

#workClasses = (:default, :unemployed, :school)
work_classes = (none=1, default=2, unemployed=3, school=4,)  # immutable tuple

function genWorkType(age_index; age_choices, p)
    if age_index == 1 #age_choices[1]
        return work_classes.none
    elseif age_index == 2 #age_choices[2]
        return work_classes.school
    elseif rand() < p.employment_rate
        return work_classes.default
    else
        return work_classes.unemployed
    end
end


# return index into employment tuple
#=
for i in 1:10
    age_indexes = rand(1:4, 100)
    println([sum(age_indexes .== i) for i in 1:4])
    work_types = genWorkType.(age_indexes; age_choices=age_choices, p=p)# , age_choices, p)
    println([sum(work_types .== i) for i in 1:4])
end
println(work_types[:])
=#


# might not work without the semi-colon (might not vectorize with the "." operator)
# returns an index [1:length(work_places]
work_types = genWorkType.(age_indexes; age_choices=age_choices, p=p)
citizen_work_types = [(work_types[i], citizens[i]) for i in 1:population]
# column vector of pairs

# Mix the students, unemployed, working ...
# Mix the work_types array
Random.shuffle!(work_types)
work_classes = (none=1, default=2, unemployed=3, school=4,)  # immutable tuple

working    = work_types[work_types .== work_classes.default]
students   = work_types[work_types .== work_classes.school]
unemployed = work_types[work_types .== work_classes.unemployed]
none_class = work_types[work_types .== work_classes.none]   # What does this correspond to?
@assert length(working) + length(students) + length(unemployed) + length(none_class) == population

class_count     = Int(ceil(length(students)/ p.class_size))
workgroup_count = Int(ceil(length(students)/ p.class_size))
environment_count  = class_count + workgroup_count  # ???
# classes
# For each students, what class are their in: (student_id, class_id)
school_assignments = [[students[i],i%class_count] for i in 1:length(students)]
work_assignments   = [[working[i], i%workgroup_count] for i in 1:length(working)]
# a 0 denotes nothing
unassigned         = [[unemployed[i], 0] for i in 1:length(unemployed)]
none_class         = [[none_class[i], 0] for i in 1:length(none_class)]

school_assignments_ = vcat(school_assignments, work_assignments, unassigned, none_class)
# Why sort?
# What is assignments for?  Graph construction?
# Sort by second index, unzip, and extract the 2nd indeo
sort_sa = sort(school_assignments_)  # sort by 1st arg of 2-tuple
# list of assignments: schools, work, unassigned
assignments = [sort_sa[i][2] for i in 1:length(sort_sa)];
#=

sims = np.array([citizens,houseNumbers,workTypes,assignments])
graph = nx.Graph()
graph.add_nodes_from(list(range(population)))
=#
sims = hcat(citizens, house_numbers, work_types, assignments)

# Graph with one node per citizen. No edges
graph = SimpleWeightedGraph(population)

# Distribute workers into different environments

# citizens connected all to all
#function to create homogeneous group
function groupCitizens!(graph, citizens, weight)
    groupSize = length(citizens)
    for i in 1:groupSize
        for j in 1:i
            #println("Edge: $i, $j")
            add_edge!(graph, citizens[i], citizens[j], weight)
        end
    end
end


#link population in the same households
# If household has 6 nodes, there are 6*5/2 = 15 connection
citizen_houses = hcat(collect(citizens), house_numbers)
for i in 1:p.households
    cond = citizen_houses[:,2] .== i-1
    house_occupants = citizen_houses[:,1][cond]
    groupCitizens!(graph, house_occupants, p.house_infectivity)
end

# Link population in the same work environment
# If work environment is too large, not everybody can be connected to everybody
# If work has 15 employees, connections are 15*16/2 = 120
# If work has 1000 employees, connections are 1000*1001/2 = 500,000 (too much)
# So we assume that somebody has no more than 10 contacts
link population in the same work environmen
assignmentGroups = list(zip(citizens, assignments))
for i in range(environmentCount):
    environmentGroup = list(zip(*list(filter(lambda x:(x[1]==i),assignmentGroups))))[0]
    groupCitizens(graph, environmentGroup, workInfectivity)


# A multilayer would allow multiple people from the same household to go to the same workplace.
# A single graph cannot have two edges between two nodes.



done

"""

#distribute workers into different environments

#link population in the same households
citizenHouses = list(zip(citizens,houseNumbers))
for i in range(houseHolds):
    house = list(zip(*list(filter(lambda x: (x[1]==i),citizenHouses))))[0]
    groupCitizens(graph, house, houseInfectivity)

#link population in the same work environmen
assignmentGroups = list(zip(citizens, assignments))
for i in range(environmentCount):
    environmentGroup = list(zip(*list(filter(lambda x:(x[1]==i),assignmentGroups))))[0]
    groupCitizens(graph, environmentGroup, workInfectivity)

nx.draw(graph)
plt.show()
for i in range(epidemicSims):
    t,S,I,R = EoN.fast_SIR(graph, globalInfectionRate, recoveryRate, rho = 0.01, transmission_weight ='transmission_weight')
    plt.plot(t,R)
    plt.plot(t,I)
    plt.plot(t,S)
plt.xlabel("time")
plt.ylabel("citizens")
plt.show()
"""
