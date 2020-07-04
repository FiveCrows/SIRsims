# Based off the latest code written by Bryan, with a network closer to demographics
using LightGraphs, SimpleWeightedGraphs, MetaGraphs, CSV
using DataFrames, Distributions, Random
using BenchmarkTools

function test()
    master_size = 300_000
    master_graph = MetaGraph(master_size)
    sz = 10_000
    mgh = nothing000

    @time for m in 1:100
        println_src(m)
        printldssrc(m)
        eight = 0.3


        #for (i,e) in enumerate(edges(mgh))
            #s, d = src(e), dst(e)
            #add_edge!(master_graph, nodes[s], nodes[d], :weights, weight)
        #end
    end
    println(ne(master_graph))
    return master_graph, mgh
end

mg, mgh = test()

function test_2()
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0
    master_size = 200_000
    master_graph = MetaGraph(master_size)
    sz = 1000
    mgh = watts_strogatz(sz, 10, 0.3)
    println("watts_strogatz: ", mgh)
    weight = 0.3
    nodes = zeros(Int64, sz)
    sample!(1:master_size, nodes, replace=false)

    function inner(mgh, mater_graph, master_size)
        for m in 1:100
            for (i,e) in enumerate(edges(mgh))
                s, d = src(e), dst(e)
                add_edge!(master_graph, nodes[s], nodes[d], :weights, weight)
            end
        end
    end

    t = inner(mgh, master_graph, master_size)

    println("--------------------")
    println("nb edges in master_graph: ", ne(master_graph))
    return master_graph, mgh
end

mg, mgh = test_2()

const master_size = 200_000
const sz = 1_000_000
const nodes_src = zeros(Int64, sz)
const nodes_dst = zeros(Int64, sz)
sample!(1:master_size, nodes_src, replace=true)
sample!(1:master_size, nodes_dst, replace=true)

function test_3()
    master_graph = MetaGraph(master_size)
    weight = 0.3

    #@time for i in 1:sz
    @time for i in 1:1000000
        s, d = nodes_src[i], nodes_dst[i]
        add_edge!(master_graph, s, d, :weights, weight)
    end
    println("--------------------")
    println("nb edges in master_graph: ", ne(master_graph))
    return master_graph
end

test_3()


function doit!(mg, ns, nd)
   for i = 1:length(ns)
      s, d = ns[i], nd[i]
      add_edge!(mg, s, d, :weights, 0.3)
   end
   return mg
end

function test_code()
    mg = MetaGraph(200_000)
    ns=rand(1:200_000, 1_000_000)
    nd=rand(1:200_000, 1_000_000)
    @time doit!(mg, ns, nd)
end

@benchmark doit!(mg, ns, nd) setup=(mg = MetaGraph(200_000); ns=rand(1:200_000, 1_000_000); nd=rand(1:200_000, 1_000_000))
@benchmark doit!(mg, ns, nd) setup=(mg = MetaGraph(200_000); ns=rand(1:200_000, 100_000); nd=rand(1:200_000, 100_000))
@benchmark doit!(mg, ns, nd) setup=(mg = MetaGraph(200_000); ns=rand(1:200_000, 10_000); nd=rand(1:200_000, 10_000))

@time doit!(mg, ns, nd) setup=(mg = MetaGraph(200_000); ns=rand(1:200_000, 1_000_000); nd=rand(1:200_000, 1_000_000))

test_code()
