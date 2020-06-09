
nb_nodes = 1000 #100000;
edges_per_vertex = 10 #30;
graph = random_regular_digraph(nb_nodes, edges_per_vertex)

function makeGraph(nb_nodes, edges_per_vertex)
    random_regular_digraph(nb_nodes, edges_per_vertex)
end

# Consumes too much memory
mutable struct Node
    index::Int  # :S, :I, :R
    status::Symbol  # :S, :I, :R
    pred_inf_time::Float64
    rec_time::Float64
    Node(index::Int, status::Symbol, pred_inf_time::Float64, rec_time::Float64) =
        new(index, status, pred_inf_time, rec_time)
end

function copy(node::Node)
    Node(node.index, node.status, node.pred_inf_time, node.rec_time)
end

# I cannot set fields if immutable
mutable struct Event
    node::Node   # is a reference
    time::Float64
    action::Symbol # :Rec, :Inf
    Event(node::Node, time::Float64, action::Symbol) =
        new(node, time, action)
end
