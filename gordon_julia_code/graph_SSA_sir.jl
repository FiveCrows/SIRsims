# Author: Gordon Erlebacher
# Date: 2020-06-06
# Create my own version of SSA that operates on graphs.
# Restrict the algorithm to first order epidemiology models

using LightGraphs
import Random: seed!
using Distributions
include("module_explorations.jl");

#my_rate = 3.0
#nb_eq_types = 2
#nb_stoich = 2
#nb_states = 3   # S, I, R for each node

#=
s = []
@time for i in 1:1000000
 dt = rand(Exponential(1/my_rate), 10)
 push!(s, dt)
end
m = mean(s)
println("mean:  $(mean(s))");
println("std:  $(std(s))");
#println("mean?:  $(sum(s)/length(s))");  # Identical to mean
mean(s)-std(s)
=#

# Generate all random numbers in advance. Much much faster
#@time dt = rand(Exponential(1/my_rate), 10000000)

#=
# same speed as rand of 10,000,000 elements
@time for i in 1:10
    dt = rand(Exponential(1/my_rate), 1000000)
end
=#

function initialData(; nb_stoich::Int, nb_nodes::Int, nb_eq_types::Int)
    println("initialData: nb_nodes= $nb_nodes")
    # Reaction equations involve two nodes
    # How to initialize arrays of a given size????
    stoch = Array{Int,nb_stoich}[]  # Could use StaticArray, but nb rates could grow
    # Hardcoded for now
    push!(stoch, [0 0 0; -1 1 0])  # S -> I
    push!(stoch, [0 -1 1; 0 0 0])  # I -> R
    β, γ = 0.5, 0.25
    rates = Array{Float64,1}([β, γ])
    eq_types = Array{Equations1.EquationType1,1}(undef, 2)
    for i = 1:nb_eq_types
        eq_types[i] = Equations1.EquationType1(nb_nodes, rates[i], stoch[i])
    end
    return eq_types, rates
end

function initializeStates(graph; nb_states)
    #@time graph = random_regular_digraph(nb_nodes, edges_per_vertex);
    #println("graph creattion\n")
    nb_edges = ne(graph)
    nb_nodes = nv(graph)

    S0 = ones(Int, nb_nodes)
    I0 = zeros(Int, nb_nodes)
    R0 = zeros(Int, nb_nodes)
    #println("S0: ", size(S0))

    function infect(i)
        S0[i] = 0 # One person is infected
        I0[i] = 1
        R0[i] = 1 - S0[1] - I0[1]
    end

    infect.([3]) #,10,15,25])
    # If S0 changes, so does statevars.states[1], which is an array of length nb_nodes
    statevars = Equations1.StateData(nb_states, [S0, I0, R0])
end

#----------------------------------------------------------------------

# For each equation type, I should store a list of equations. These equations
# should take the form of a list of node pairs, i.e., a set of edges.
# This means that one must scan the entire list of edges nb_types times, where
# nb_types equals the number of different rates.
# The alternative is to scan the different types for each edge. For cache reasons,
# it is better that the shortest loops be the outer loops.


function update(
    ktype::Int,
    node1::Int,
    node2::Int,
    eq_type,
    eqs,
    nb_states::Int,
)
    # Update equation type "ktype", and equation "jeq" within that type
    # 4 then 10 allocations: 224 bytes per loopo. Why?
    # about 6M of allocations per loop. WHY!!!
    #println("==> enter update: fran")
    for s = 1:nb_states
        eqs[s][node1] += eq_type.Δstoich[1,s]  # 56 allocs
    end
    for s = 1:nb_states
        eqs[s][node2] += eq_type.Δstoich[2, s]
    end
    # without this line, there is losts of allocation in order to return data
    nothing
end


# Some timings to test graph access
# @time Equations1.tst()

# 8.5sec, alloc: 3.2GB
function updateAll()
    # graph creation: 3M edges: 1.5 sec
    nb_nodes = 100000 #100000;
    edges_per_vertex = 30 #30;
    graph = random_regular_digraph(nb_nodes, edges_per_vertex)

    nb_states = 3
    nb_eq_types = 2

    eq_types, rates = initialData(nb_stoich=2, nb_nodes=nb_nodes, nb_eq_types=nb_eq_types)
    #println("eq_types= ", eq_types)
    state_vars = initializeStates(graph, nb_states=nb_states)
    #println(typeof(state_vars), size(state_vars.states))

    # Loop on large graph over 3M edges: 0.2 sec. Could be parallelized
    ktype in 1:nb_eq_types
        eq_type = eq_types[ktype] # no allocaitons
        eqs = state_vars.states   # no allocaitons

        for e in edges(graph)    # .24sec per loop
            node1 = src(e)
            node2 = dst(e)
            update(ktype, node1, node2, eq_type, eqs, nb_states)
        end
    end
    return graph
end

# Test updating every edge once. Very fast.
@time updateAll()
#----------------------------------------------------------------------
function simulate()
    # Initialie Global times and scaled times for each type
    nb_nodes = 100000 #100000;
    edges_per_vertex = 30 #30;
    @time graph = random_regular_digraph(nb_nodes, edges_per_vertex)
    print("Completed graph creation\n")
    nb_edges = ne(graph)
    nb_states = 3  # hard-coded for now
    nb_eq_types = 2  # hard-coded for now
    state_vars = initializeStates(graph, nb_states=nb_states)
    eq_types, rates = initialData(nb_stoich=2, nb_nodes=nb_nodes, nb_eq_types=nb_eq_types)
    t_max = 10000000.  # total simulation time
    eqs = state_vars.states   # no allocaitons
    println()
    println("rates= $rates")

    edge_nodes = zeros(Int,2,nb_edges)

    @time for (i,e) in enumerate(edges(graph))
        edge_nodes[1,i] = src(e)
        edge_nodes[2,i] = dst(e)
    end

    function updatePropensities!(propensities::Array{Float64})
        # state variables are defined on the nodes
        eqs = state_vars.states   # no allocaitons
        for n in 1:nb_eq_types    # .24sec per loop
            propensities[n] = rates[n]
        end
    end

    function inner(graph)
        t_glob = 0.0
        t_types = zeros(nb_eq_types)
        s_types = zeros(nb_eq_types)
        #println(t_types)

        for i in 1:nb_eq_types
            s_types[i] = rand(Exponential(1.))
        end

        # (3) Calculate propensities for all equations

        count = 0

        while true
            count += 1
            propensities = zeros(nb_eq_types)
            updatePropensities!(propensities)

            # rescale time to next raction
            # The method asks that a time be generated for each equation.
            # But because state variables are either 1 or zero, I feel that
            # there is only a need to generate it once per type.
            Δt_types = (s_types .- t_types) ./ propensities
            Δtmin = minimum(Δt_types)
            eq_type = argmin(Δt_types)
            #println("eq_type= $eq_type")
            #println("Δt_types= $Δt_types")
            #println("Δtmin= $Δtmin")

            #println("t_glob= $t_glob")
            if (t_glob+Δtmin) > t_max
                # End simulation
                println("End Simulation, maximum time reached.")
                print("nb iterations: ", count)
                nothing
                return
            end

            t_glob += Δtmin

            # Update rescaled times for each type
            t_types .+= (propensities[eq_type] * Δt_types[eq_type])

            # Choose a random equation of type eq_type and udpate it
            if eq_type == 1
                # I know thia is  S -> R, defined on edges. But how to automate this?
                #  User should provide a function somehow
                edge_index = rand(1:nb_edges)
                node1 = edge_nodes[1, edge_index]
                node2 = edge_nodes[2, edge_index]
            else
                node1 = rand(1:nb_nodes)
                node2 = node1
            end

            update(eq_type, node1, node2, eq_types[eq_type], eqs, nb_states)

            # Generate scaled next reaction time
            Δs = zeros(2)
            Δs[eq_type] = rand(Exponential(1.))
            # update next scaled reaction type for reaction type eq_type
            s_types[eq_type] += Δs[eq_type]
        end

    end

    @time inner(graph)
    println("Completed inner()\n")

    nothing
end

@time simulate()
