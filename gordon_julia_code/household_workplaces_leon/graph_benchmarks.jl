# Add and remove random edges from a graph and perform some benchnarking
# as a function of graph size

using LightGraphs
using BenchmarkTools
using DataFrames
using Plots
#using StringLiterals

function generateGraphs()
    graphs = []
    nnodes = [1000, 10000, 100000]
    edge_scale = [5, 10, 20]
    for i in 1:length(nnodes)
        for j in 1:length(edge_scale)
            push!(graphs, SimpleDiGraph(nnodes[i], nnodes[i]*edge_scale[j]))
        end
    end
    return graphs
end

graphs = generateGraphs()

edge_lists = []
randEdgeIndexes(nb_edges, nb_rand) = rand(1:nb_edges, nb_rand)
randEdgeNodes(nb_nodes, nb_rand) = rand(1:nb_nodes, 2*nb_rand)

function runGraph(n::Int64, graph, edge_list, e_indexes, en_indexes, nb_e)
    for i in 1:n
       #rem_edge!(graph, edge_list[e_indexes[i]])  # no memory
        # 10x more expensive!
       add_edge!(graph, en_indexes[2*i-1], en_indexes[2*i])
    end
    nb_e[1] = ne(graph) # a reference
    return
end

# Estimate average time per add/remove edge pair
counts = [1000, 100, 10, 10000, 100000]
counts = [1000, 100, 10]
function runBenchmarks()
    dict = Dict()
    #for g in graphs[1:9]
    for g in graphs[4:4]
        @show g
        edge_indexes = randEdgeIndexes(ne(g), n)
        edge_node_indexes = randEdgeNodes(nv(g), n)
        edge_list = collect(edges(g))
        for n in counts
            @show n
            t = @benchmark runGraph(n, $(deepcopy(g)), $edge_list, $edge_indexes, $edge_node_indexes, $([0])) evals=1 samples=20
            dict[(g,n)] = t
        end
    end
    return dict
end

# Adding edges is 30 to 40x more expansive and removing them.
dict = runBenchmarks()

function setupDataFrame!(dict)
    gnv_l = []
    gne_l = []
    tmin_l = []
    n_l = []
    ne_end_l = []
    for key in keys(dict)
        g = key[1]
        n = key[2]
        #ne_end = key[3]
        t = dict[key]
        gnv = nv(g)
        gne = ne(g)
        # time per pair of edges added/rmeoved
        tmin = minimum(t).time / (1000*n) # μs
        println("minimum(t): ", minimum(t))
        println("minimum(t): ", dump(minimum(t)))
        tmax = maximum(t).time / (1000*n) # μs
        push!(gnv_l, gnv)
        push!(gne_l, gne)
        push!(tmin_l, tmin)
        push!(n_l, n)
        #push!(ne_end_l, ne_end)
    end
    #df = DataFrame([n_l, gnv_l, gne_l, ne_end_l, tmin_l],
    df = DataFrame([n_l, gnv_l, gne_l, tmin_l],
                   [:n, :nv,   :ne,    :tmin])
end

df = setupDataFrame!(dict)
println(df[:,:])

scatter(df.nv, df.tmin, markersize=10)
scatter(df.ne, df.tmin, markersize=10)
scatter(df.n, df.tmin, markersize=10)
df.log = log.(df.tmin)
scatter(log.(df.n), df.log)

# ----------------------------------------------------
# Benchmarks without using functions

n = 10
g = SimpleDiGraph(300000, 1000000)
edge_indexes = randEdgeIndexes(ne(g), n)
edge_node_indexes = randEdgeNodes(nv(g), n)
edge_list = collect(edges(g))
t = @benchmark runGraph(n, $(deepcopy(g)), $edge_list, $edge_indexes, $edge_node_indexes, $([0])) evals=1 samples=20
tim = minimum(t).time/(1000*n)
timtot = minimum(t).time/1_000_000_000
println("tim/edge = $tim μs, total time: $timtot s")
nothing

randEdgeIndexes(nb_edges, nb_rand) = rand(1:nb_edges, nb_rand)
randEdgeNodes(nb_nodes, nb_rand) = rand(1:nb_nodes, 2*nb_rand)

function myBenchmark(n::Int64)
    g = SimpleDiGraph(300000, 1000000)
    edge_indexes = randEdgeIndexes(ne(g), n)
    edge_node_indexes = randEdgeNodes(nv(g), n)
    edge_list = collect(edges(g))
    # The last argument of runGraph was meant to return the number of edges in the graph after the benchmark
    t = @benchmark runGraph($n, $(deepcopy(g)), $edge_list, $edge_indexes, $edge_node_indexes, $([0])) evals=1 samples=20
    tim = minimum(t).time/(1000*n)
    timtot = minimum(t).time/1000000000
    println()
    println("tim/edge = $tim μs, total time: $timtot s")
    return
end

myBenchmark(10)
myBenchmark.([10, 100, 1000, 10000, 100000])
