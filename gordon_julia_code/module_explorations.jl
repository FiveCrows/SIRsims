module Equations1
using LightGraphs

mutable struct EquationType2
    neq::Int
    rate::Float64
   Δstoich::Int
   EquationType2(neq::Int, rate::Float64,stoich::Int) = new(neq, rate, stoich)
   # Δstoich::Array{Int, 1}  # one triplet per node  for each edge
   # EquationType2(neq::Int, rate::Float64,stoich::Array{Int,1}) = new(neq, rate, stoich) end
end

mutable struct EquationType1
    neq::Int
    rate::Float64
    Δstoich::Array{Int, 2}  # one triplet per n[[1,1,1],[-1,1,-1]]ode  for each edge
    EquationType1(neq::Int, rate::Float64,stoich::Array{Int,2}) = new(neq, rate, stoich)
end

# Store the state data (SIR) for each node of the graph
# Ordered by nodex index.
#    Higher efficiency might be possible via appropriate reordering.
mutable struct StateData
    nb_state_vars::Int
    states::Array{Array{Int,1},1}
    StateData(nb_vars::Int, state_arrays::Array{Array{Int,1},1}) =
        new(nb_vars, state_arrays)
end

# Timngs for graph with 3 million edges
# Prestore all the edge nodes
# 2 second for loop over 3,000,000 edges. Why?
function tst()
    nb_nodes = 100000;
    edges_per_vertex = 30
    graph = random_regular_digraph(nb_nodes, edges_per_vertex);
    nb_edges = ne(graph)
    nb_nodes = nv(graph)
    nodes1 = zeros(Int,2,nb_edges)
    nodes2 = zeros(Int,2,nb_edges)
    nodes3 = zeros(Int,nb_edges,2)
    nodes4 = zeros(Int,nb_edges,2)

    println("\n\n------------------- TIMINGS --------------\n\n")
    println("nodes1 = zeros(Int,2,nb_edges)")
    println("nodes2 = zeros(Int,2,nb_edges)")
    println("nodes3 = zeros(Int,nb_edges,2)")
    println("nodes4 = zeros(Int,nb_edges,2)")

    println("--------\n")
    @time for (i,e) in enumerate(edges(graph))
        nodes3[i,1] = src(e)
        nodes4[i,2] = dst(e)
    end
    println("Edge loop, nodes3, nodes4\n")

    println("Loop over nodes completed\n")
    @time for (i,e) in enumerate(edges(graph))
        nodes1[1, i] = src(e)
        nodes1[2, i] = dst(e)
    end
    println("Loop over nodes completed\n")
    @time adj = adjacency_matrix(graph)
    println("adjacency_matrix completed\n")

    @time for j in 1:2
        for i in 1:nb_edges
            nodes2[j,i] = nodes1[j,i]
        end
    end
    println("Not cache friendly, nodes2[i,j] = nodes1[i,j], double loop\n")

    @time for i in 1:nb_edges
        for j in 1:2
            nodes2[j,i] = nodes1[j,i]
        end
    end
    println("Cache friendly, nodes2[i,j] = nodes1[i,j], double loop\n")

    @time for i in 1:nb_edges
    for j in 1:2
            nodes4[i,j] = nodes3[i,j]
        end
    end
    println("Not cache friendly, nodes4[i,j] = nodes1[i,j], double loop\n")

    @time for j in 1:2
        for i in 1:nb_edges
            nodes4[i,j] = nodes3[i,j]
        end
    end
    println("Cache friendly, nodes4[i,j] = nodes1[i,j], double loop\n")

    # Loop over nodes
    @time nodes1 = nodes1
    println("nodes1 = nodes1\n")
    @time nodes1 = copy(nodes1) # 2x speed of double loop
    println("nodes1 = copy(nodes1)\n")
    @time nodes1 = deepcopy(nodes1) # 1/3 speed of double loop
    println("nodes1 = deepcopy(nodes1)\n")
    println("INside Exploration Module")
end


# END module
end
