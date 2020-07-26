using LightGraphs
using DataFrames
using Random
using PyCall
@pyimport pickle

# Given two subgraphs, I would like to connect them.
# Assign two ages, 5 and 10 to each node
n1 = 200000
n2 = 100000
nb_nodes = n1 + n2
g = SimpleGraph(nb_nodes, 0)

# nb connections between the two groups: 600
nb_conn = 600000
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

function setupGraph(n1, n2)
    nb_nodes = n1 + n2
    g = SimpleGraph(nb_nodes, 0)
    return g
end

# Use dictionaries for speed
function edgeList2(n1,deg1,n2,deg2)

    println("============================")
    function setup(n1,deg1,n2,deg2)
        n = n1 + n2
        deg_dict = Dict()
        deg_list = []
        edge_dict = Dict()
        excess_list = []
        @show deg1, deg2, n1, n2
        # These sequences made things more expensive!
        seq1 = repeat(collect(1:n1), deg1)
        seq2 = repeat(collect(n1+1:n1+n2), deg2)
        seq1 = seq1[randperm(length(seq1))]
        seq2 = seq2[randperm(length(seq2))]
        seq = hcat(seq1, seq2)
        @show length(seq1)
        return deg_dict, deg_list, edge_dict, excess_list, seq1, seq2, seq
    end
    deg_dict, deg_list, edge_dict, excess_list, seq1, seq2, seq = setup(n1,deg1,n2,deg2)

    #deg_list = zeros(Int8,length(seq1))
    #for i in 1:length(seq1)
        #deg_dict[i] = 0
        #deg_list[i] = 0
    #end

    @time for i in 1:length(seq1)  # nb of edges to add
        e1 = seq[i,1]
        e2 = seq[i,2]
        mn = minimum((e1,e2))
        mx = maximum((e1,e2))

        if haskey(edge_dict,(mn,mx))
            push!(excess_list, (mn, mx))
            continue
        end

        #deg_dict[i] = 1
        edge_dict[(mn,mx)] = 1

    end
    @show length(excess_list)
    @show length(edge_dict)
    println("percentage edges missing: ", 100*length(excess_list)/length(edge_dict), " percent")
    return edge_dict, excess_list
end


### VERY FAST. 0.02 sec for a graph of 6000 edges.
### Assume 10000 graphs of this size: 0.02 * 10000 = 200 sec. VEREY HIGH COST.
### Why is it that

n1 = 200_000
n2 = 100_000
deg1 = 3
deg2 = 6
g = setupGraph(n1, n2)

# I MUST REDUCE ALLOCATIONS!! 1.6 seconds for 600,000 edges
@time edge_dict, excess_list = edgeList2(n1, deg1, n2, deg2)
#@btime edge_dict, excess_list = edgeList2(n1, deg1, n2, deg2) seconds=5

# ----------------------------------------
# Create two degree pdfs, one for each subgraph, and establish connections such
# the degree distributions match the pdfs.
# So assume pdf1, and for each i in [0,N1], d_i \in pdf1  (discrete distribution)
# Keep sampling until \sum d_i = S_1  (= sum of degrees in V1).
# Similarly, calculate \sum d_i = S_2 (= sum of degrees in V2)m where d_i ~ pdf2.
# Ideally, S_1 == S_2. If that is not the case, WHAT TO DO? We know that the pdfs
# are chosen so that the average degrees satisfy: N1*<d1> = N2*<d2> where d1 is the
# collection of degrees in V1, and similarly for d2. Let us try this out and see what happens.
# ----------------------------------------

# Use dictionaries for speed
# Use probability density functions for the degrees to handle
# non-integer averages
function edgeListProb(n1,deg1,n2,deg2)

    println("============================")
    function setup(n1,deg1,n2,deg2)
        n = n1 + n2
        deg_dict = Dict()
        deg_list = []
        edge_dict = Dict()
        excess_list = []
        @show deg1, deg2, n1, n2
        # These sequences made things more expensive!
        seq1 = repeat(collect(1:n1), deg1)
        seq2 = repeat(collect(n1+1:n1+n2), deg2)
        seq1 = seq1[randperm(length(seq1))]
        seq2 = seq2[randperm(length(seq2))]
        seq = hcat(seq1, seq2)
        @show length(seq1)
        return deg_dict, deg_list, edge_dict, excess_list, seq1, seq2, seq
    end
    deg_dict, deg_list, edge_dict, excess_list, seq1, seq2, seq = setup(n1,deg1,n2,deg2)

    #deg_list = zeros(Int8,length(seq1))
    #for i in 1:length(seq1)
        #deg_dict[i] = 0
        #deg_list[i] = 0
    #end

    @time for i in 1:length(seq1)  # nb of edges to add
        e1 = seq[i,1]
        e2 = seq[i,2]
        mn = minimum((e1,e2))
        mx = maximum((e1,e2))

        if haskey(edge_dict,(mn,mx))
            push!(excess_list, (mn, mx))
            continue
        end

        #deg_dict[i] = 1
        edge_dict[(mn,mx)] = 1

    end
    @show length(excess_list)
    @show length(edge_dict)
    println("percentage edges missing: ", 100*length(excess_list)/length(edge_dict), " percent")
    return edge_dict, excess_list
end


n1 = 373
n2 = 693
deg1 = 4
deg2 = 6
# We want 2000 links between both groups
# Create categorical probabilities
# m1 with deg 3, m2 with deg 4, m3 with deg 5
# <d1> = 2000/373 = 5.36
# <d2> = 2000/693 = 2.89
# d11*q1 + d12*q2 = 6000  (q for V1)
# eg.   5 * q1 + 6 * q2 = 5.36
#  and q1 + q2 = 1
# Therefore: 5 * q1 + 6 * (1-q1) = -q1 + 6 = 5.36 ==> q1 = 0.64
# Categorical probability is 5 * 0.64 + 6 * 0.36 = 5.36
# So with two categories, I can create a pdf.
# So now to sample from this pdf. How many samples do I need to achieve
# the proper average.

using Distributions

# Generate a sequence of degrees that match a particular average.
# This is for the case when the average degree is non-integer
function generateSequence(N::Int64, avg_deg)
    d1 = Int8(floor(avg_deg))
    d2 = d1 + 1
    # d1 * q1 + d2 * q2 = avg
    # q1 + q2 = 1
    # d1 * q1 + d2 * (1-q1) = avg
    # q1 * (d1 - d2) + d2 = avg
    # q1 = (avg - d2) / (d1 - d2)
    # q2 = 1 - q2
    q1 = (d2 - avg_deg)
    q2 = 1. - q1
    catt = Distributions.Categorical([q1, q2])
    vals = (d1, d2)
    r = rand(catt, N) # 50% of time
    degrees = zeros(Int8, length(r))
    for i in 1:length(r)   # 50% of time
        degrees[i] = vals[r[i]]
    end
    #@time degrees = [vals[r[i]] for i in 1:length(r)]
    #println("mean degree approx: $(mean(degrees))")
    return degrees, mean(degrees)
end

n1 = 373
n2 = 693
nb_edges = 2000

avg_deg1 = nb_edges / n1
avg_deg2 = nb_edges / n2
# We want 2000 links between both groups
@time seq1, avg_deg1_approx = generateSequence(n1, avg_deg1)
@time seq2, avg_deg2_approx = generateSequence(n2, avg_deg2)
# Repeat each node a number of times equal to its degree (There has to be a better approach)
# duplicte nodes
# Because the degrees are only of two values, separated by one, there is a fast algorithm.
deg1 = Int8(floor(avg_deg1))
nodes1 = collect(1:length(seq1))
nodes11 = nodes1[seq1 .== deg1]
nodes12 = nodes1[seq1 .== (deg1+1)]
hh1 = vcat(repeat(nodes11, deg1), repeat(nodes12, deg1+1))

deg2 = Int8(floor(avg_deg2))
nodes2 = collect(1:length(seq2))
nodes21 = nodes2[seq2 .== deg2]
nodes22 = nodes2[seq2 .== (deg2+1)]
hh2 = vcat(repeat(nodes21, deg2), repeat(nodes22, deg2+1))

total_deg1 = sum(seq1)
total_deg2 = sum(seq2)

# Given hh1 and hh2 of different lengths, establish the links between subgraphs

#g = setupGraph(n1, n2)
#@time edge_dict, excess_list = edgeList2(n1, deg1, n2, deg2)

# Try this out. Let us connect the two subgraphs

# Establish the connections between two subgraphs that satisfy a given
# categorical distribution. We model the distribution with two entries such
# that the degree average is correct. The degree averages of the two subgraphs
# are nb_edges/n1 and nb_edges/n2, which are not integer values in general.
# nb_edges: desired total number of edges between two subgraphs
# n1, n2: number of nodes in each subgraph
function edgeListProb(n1, n2, nb_edges)
    #function setup(n1,deg1,n2,deg2)
        edge_dict = Dict()
        excess_list = []

    avg_deg1 = nb_edges / n1
    avg_deg2 = nb_edges / n2
    @show avg_deg1, avg_deg2
    # We want 2000 links between both groups
    seq1, avg_deg1_approx = generateSequence(n1, avg_deg1)
    seq2, avg_deg2_approx = generateSequence(n2, avg_deg2)
    # seq1 and seq2 = list of degrees
    # Repeat each node a number of times equal to its degree (There has to be a better approach)
    # duplicte nodes
    # Because the degrees are only of two values, separated by one, there is a fast algorithm.
    deg1 = Int16(floor(avg_deg1))
    nodes1 = collect(1:length(seq1))
    nodes11 = nodes1[seq1 .== deg1]
    nodes12 = nodes1[seq1 .== (deg1+1)]
    hh1 = vcat(repeat(nodes11, deg1), repeat(nodes12, deg1+1))

    deg2 = Int16(floor(avg_deg2))
    lg = length(seq1)
    nodes2 = collect(lg+1:lg+length(seq2))
    nodes21 = nodes2[seq2 .== deg2]
    nodes22 = nodes2[seq2 .== (deg2+1)]
    hh2 = vcat(repeat(nodes21, deg2), repeat(nodes22, deg2+1))

    hh1 = hh1[randperm(length(hh1))]
    hh2 = hh2[randperm(length(hh2))]

    total_deg1 = sum(seq1)
    total_deg2 = sum(seq2)

    # Given hh1 and hh2 of different lengths, establish the links between subgraphs

    lg = minimum((length(hh1), length(hh2)))
    lgmax = maximum((length(hh1), length(hh2)))
    edge_dict = Dict()
    for i in 1:lg
        e1 = hh1[i]
        e2 = hh2[i]
        mn = minimum((e1,e2))
        mx = maximum((e1,e2))

        if haskey(edge_dict,(mn,mx))
            push!(excess_list, (mn, mx))
            continue
        end

        edge_dict[(mn,mx)] = 1
    end
    #@show count, lg
    #@show length(edge_dict)
    #@show length(excess_list)
    @assert length(edge_dict) + length(excess_list) == lg
    #@show avg_deg1, avg_deg2
    #@show 100*length(excess_list) / lgmax
    return edge_dict, excess_list
end

# Create a test
function tests()
    # graph sizes
    nb_graphs = 100
    nb_nodes1 = rand(200:500, nb_graphs)
    nb_nodes2 = rand(200:500, nb_graphs)
    gl = Array{SimpleGraph, 1}()
    for n in 1:length(nb_nodes1)
        push!(gl, SimpleGraph(nb_nodes1[n]+nb_nodes2[n]))
    end
    # Assume constant average degrees avg_deg1 and avg_deg2
    # There is a problem. If n1, n2 are given, as well as nb_edges,
    # this imposes an average degree on both sides. So how are contact
    # matrices dealt with? If I have two age groups of size n1 and n2,
    # and the contact matrix states that there are N1 total contacts
    # from group 1 to group 2 and 5.2 average contacts from group 2
    # to group 1
    avg_deg1 = 6.7
    avg_deg2 = 10.4

    for (i,g) in enumerate(gl[1:100])
        println("==> i= $i")
        n1 = nb_nodes1[i]
        n2 = nb_nodes2[i]
        # Total contacts between the two groups is a
        navg = 0.5*(n1+n2)
        scale = rand(5:10, 1)
        nb_edges = navg*scale[1]
        #@show nb_edges, n1, n2
        #show nb_edges/n1, nb_edges/n2
        edge_dict, excess_list = edgeListProb(n1, n2, nb_edges)

        for k in keys(edge_dict)
            add_edge!(g, k[1], k[2])
        end

        d = degree(g)
        @show final_avgd1 = sum(d[1:n1]) / n1
        @show final_avgd2 = sum(d[1+n1:n1+n2]) / n2
    end
end

@time tests()

n1 = 15373
n2 = 8952
nb_edges = 60000
@time edge_dict, excess_list = edgeListProb(n1, n2, nb_edges)

g = SimpleGraph(n1+n2)
for k in keys(edge_dict)
    add_edge!(g, k[1], k[2])
end

@show Δ(g)
d = degree(g)
@show avgd1 = sum(d[1:n1]) / n1
@show avgd2 = sum(d[1+n1:n1+n2]) / n2

import CSV
df = CSV.read("school_contact_matrix.csv")
arr = df[!, "450145980"]
arr = Vector(arr)
arr = reshape(arr, 16, 16)


function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

dir = "ContactMatrices/Leon/"
ds = myunpickle(dir*"ContactMatrixSchools.pkl")
dw = myunpickle(dir*"ContactMatrixWorkplaces.pkl")

d = ds[450122676]
