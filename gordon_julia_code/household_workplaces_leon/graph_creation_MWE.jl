# Based off the latest code written by Bryan, with a network closer to demographics
using LightGraphs, SimpleWeightedGraphs, MetaGraphs, CSV
using DataFrames, Distributions, Random

# ----------------------------------------------------------------------
function myMerge!(master_graph::MetaGraph, meta_graph::MetaGraph, weight::Float64)
    n = get_prop(meta_graph, :nodes)
    for e in edges(meta_graph)
        i = src(e)
        j = dst(e)
        add_edge!(master_graph, n[i], n[j], :weight, weight)
    end
end
# ----------------------------------------------------
function createWorkGraph(master_size, groups, β_strogatz, weight::Float64)
    elapsed_setup = zeros(15000)
    elapsed_metagraph = zeros(15000)
    elapsed_merge = zeros(15000)

    master_graph = MetaGraph(master_size)

    for (i,grp) in enumerate(groups)
        elapsed_setup[i] = @elapsed begin
            person_ids = grp
            sz = length(person_ids)
            if i % 100 == 0
                println("i,sz: $i, $sz")
            end
            if sz > 5000
                println("###################################")
                println("sz > 5000 (= $sz), SKIP OVER")
                println("###################################")
                continue
            end
        end
        elapsed_metagraph[i] = @elapsed begin
            if sz == 1
                continue
            elseif sz < 5
                mgh = createDenseGraph(person_ids) #p::Float64)
            elseif sz < 25
                mgh = createErdosRenyiGraph(person_ids, 10) #p::Float64)
            elseif sz < 100
                mgh = watts_strogatz(sz, 15, β_strogatz) # erdos-redyi
            else
                mgh = watts_strogatz(sz, 10, β_strogatz)
            end
        end

        if sz > 1
            elapsed_merge[i] = @elapsed begin
                mg  = MetaGraph(mgh, weight)
                set_prop!(mg, :nodes, person_ids)
                myMerge!(master_graph, mg, weight)
            end
        end
    end
    println("setup time elapsed: ", sum(elapsed_setup))
    println("metagraph time elapsed: ", sum(elapsed_metagraph))
    println("merge time elapsed: ", sum(elapsed_merge))
    return master_graph
end
# ----------------------------------------------------
function createErdosRenyiGraph(group, m::Int) #p::Float64)
    n = length(group)
    p = 2*m / (n-1)
    graph = erdos_renyi(n, p)
end
# ----------------------------------------------------
# 1) create a dense graph, given a group
function createDenseGraph(group)
    n = length(group)
    # add Metadata to the nodes and edges with the person_id, or a unique id in 1:population_size
    graph = SimpleGraph(n, div(n*(n-1),2))
end
# ----------------------------------------------------
function generateDemographicGraphs()

    # Create a fake dataframe with columns :person_id, :work_id, :school_id
    # I want 60 schools, 10,000 workplaces, 250,000 people
    nb_people = 250000
    nb_businesses = 10000
    work_sizes = [3, 70, 300, 5000]
    work_prob = [.7, .12, .09, .05]
    work_prob ./=  sum(work_prob)
    categ = Distributions.Categorical(work_prob)

    work_ids   = collect(1:nb_businesses)
    work_classes = rand(categ, nb_businesses)

    work_id = rand(work_classes, nb_people)
    employees = randperm(nb_people)

    persons = collect(1:nb_people)
    # Create groups
    work_groups = []
    for i in 1:nb_businesses
        work_class = work_classes[i]
        business_size = work_sizes[work_class]
        employees = sample(persons, business_size, replace=true)
        push!(work_groups, employees)
    end
    println("nb_businesses: $nb_businesses")

    # There are actually 10,000 groups
    work_groups = work_groups[1:1000]

    tot_people = sum(map(length,work_groups))

    println("Start work")
    cwgraph = @elapsed work_graph = createWorkGraph(tot_people, work_groups, 0.31, 0.8)
    println("Finished work")

    println("total people: ", tot_people)
    println(" time to execute createWorkGraph: $cwgraph")

    return work_graph
end
# ----------------------------------------------------------------------
@time work_graph = generateDemographicGraphs()
