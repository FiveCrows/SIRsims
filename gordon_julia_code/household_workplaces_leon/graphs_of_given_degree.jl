using LightGraphs
using DataFrames
using Random

# Given two subgraphs, I would like to connect them.
# Assign two ages, 5 and 10 to each node
n1 = 20000
n2 = 10000
nb_nodes = n1 + n2
g = SimpleGraph(nb_nodes, 0)

# nb connections between the two groups: 600
nb_conn = 60000
deg1 = div(nb_conn, n1)   # degree of nodes in V1
deg2 = div(nb_conn, n2)   # degree of nodes in V2

# Create two sequences, one for V and the other for W (W + V = g)
seq1 = repeat(collect(1:n1), deg1)
seq2 = repeat(collect(n1+1:n1+n2), deg2)
seq1 = seq1[randperm(length(seq1))]
seq2 = seq2[randperm(length(seq2))]

@assert(length(seq1) == length(seq2))

seq = hcat(seq1, seq2)



function edgeList!(n1,deg1,n2,deg2)
    edge_dict = Dict()
    excess_list = []
    # These sequences made things more expensive!
    seq1 = repeat(collect(1:n1), deg1)
    seq2 = repeat(collect(n1+1:n1+n2), deg2)
    seq1 = seq1[randperm(length(seq1))]
    seq2 = seq2[randperm(length(seq2))]
    seq = hcat(seq1, seq2)

    for i in 1:length(seq1)
        e1 = seq[i,1]
        e2 = seq[i,2]
        try
            edge_dict[(e1,e2)] += 1
            push!(excess_list, minimum((e1,e2)) => maximum((e1,e2)))
        catch
            edge_dict[(e1,e2)] = 1
        end
    end
    return edge_dict, excess_list
end

#create connections
function exEdgeList!(edge_dict, old_excess_list)
    seq1 = []
    seq2 = []
    excess_list = []
    for e in old_excess_list
        push!(seq1, e.first)
        push!(seq2, e.second)
    end
    seq1 = seq1[randperm(length(seq1))]
    seq2 = seq2[randperm(length(seq2))]
    seq = hcat(seq1, seq2)    # 2 columns
    for i in 1:length(seq1)
        e1 = seq[i,1]
        e2 = seq[i,2]
        try
            edge_dict[(minimum((e1,e2)), maximum((e1,e2)))] += 1
            push!(excess_list, minimum((e1,e2)) => maximum((e1,e2)))
        catch
            edge_dict[(e1,e2)] = 1
        end
    end
    return excess_list
end

# Takes 8 sec on a graph with 60,000 nodes. VERY EXPENSIVE!!
@time edge_dict, excess_lis = edgeList!(n1,deg1,n2,deg2)
@time ex = exEdgeList!(edge_dict, excess_list)

# Much faster than unique
@time unique(edge_list)
# How to find the missing four elements?

g = SimpleGraph(nb_nodes, 0)
@time for i in 1:length(seq1)
    #println("$i, $(edge_list[i].first), $(edge_list[i].second)")
    add_edge!(g, edge_list[i].first, edge_list[i].second)
end

# Find duplicate edges
