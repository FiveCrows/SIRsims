module FastSIRMultiplex

using StaticArrays
using LightGraphs
import Random: seed!
using Distributions
using MetaGraphs
using DataStructures

# Consumes too much memory
# Everybody has state 1
# Uniform structure
# Some people have state 1,2
# Other people have state 1,3
# Still others have state 1,1 (stay at home)
# Ideally, it would be nice to have variability in terms of state changes
# This could be done with Markov for more generality
mutable struct Node
    index::Int  # :S, :I, :R
    status::Symbol  # :S, :I, :R
    pred_inf_time::Float64
    rec_time::Float64
	# We can make this more efficient at a later time
	# the state will flip between the first and the second
end

# Make sure Basic is in the path. Consider LOAD_PATH
#import Basic: copy
#function Base.copy(node::Node)
    #Node(node.index, node.status, node.pred_inf_time, node.rec_time)
#end

# I cannot set fields if immutable
mutable struct Event
    node::Node   # is a reference
    time::Float64
    action::Symbol # :Rec, :Inf
end

# get the status every time a new event is extracted to make sure I am
# propagating on a new graph
# Or better: only get status each time a infection is about to check his neighbors
function getStatus(t, node_status)
	tmod = t % 1.
	if (tmod < 0.25) || (tmod > 0.75)  # measured in days
	    return 1  # home
	end
	return node_status[2]  # school or work
end

# Every node must encode status time intervals. Some nodes
# are always at home. Some nodes go between home and school, other
# nodes go between home and work


# Replace G by Glist = [Ghome, Gschool, Gwork]
function fastSIR(Glist, params, initial_infecteds::Vector{Int}; t_max=10.)
	status = 1  # status = 1 (home), 2, school, 3, work
	global_time = 0.0  # new variable
	τ = params.τ
	γ = params.γ
	# These can no longer be precomputed because in principle, each
	# edge can have a different weight.

	G = Glist[1]  # first list is households

    nb_nodes = nv(G)
	times = [0.]
    # length(G) is Int128
    S = [nb_nodes]
    I = [0]
    R = [0]
    pred_inf_time = 100000000.0
    rec_time = 0.

	# empty queue
    Q = PriorityQueue(Base.Order.Forward);

	# prestore one F.Node for each graph vertex
    nodes = Array{Node,1}(undef, nb_nodes)
	# person status
	# I do not know how to handl non-mutable static vectors
	p_status = Vector{MVector{2, Int8}}(undef, nb_nodes)
	for i in 1:nb_nodes
		p_status[i] = (0,0)
	end

    for u in 1:nb_nodes  # extremely fast
        nodes[u] = Node(u, :S, pred_inf_time, rec_time)
		p_status[u][1] = 1
		p_status[u][2] = 1
    end

	# I need the actual person id for these graphs.
	ids = get_prop(Glist[2], :node_ids)
	for u in ids
		p_status[u][2] = 2
	end

	ids = get_prop(Glist[3], :node_ids)
	for u in ids
		p_status[u][2] = 3
	end

	println("finished with setting status")

	# REMOVE
	#println(">>> about to test")
	#testTimings(G, nodes[4])  # EXPERIMENTAL. JUST FOR TESTING REMOVE WHEN DONE
	#println("END TEST TIMINGS")
	# REMOVE

    for u in initial_infecteds
       #println("nodes[u]= ", nodes[u])
       event = Event(nodes[u], 0., :transmit)
       # How to do this with immutable structure
       nodes[u].pred_inf_time = 0.0  # should this new time be reflectd in the event? Here, it is.
       Q[event] = event.time
    end

    while !isempty(Q)
        event = dequeue!(Q)
        if event.action == :transmit
            if event.node.status == :S
                processTransSIR(Glist, p_status, event.node, event.time, τ, γ, times,
					S, I, R, Q, t_max, nodes)
            end
		else
             # 1 alloc: 16 bytes, 0.000001 to 0.000002 seconds
             processRecSIR(event.node, event.time, times, S, I, R)
             #println("processRecSIR\n")
        end
    end
	println("times, times[end]: $(length(times)), $(times[end])")
    times, S, I, R
end;


function processTransSIR(Glist, p_status, node_u, t::Float64, τ::Float64, γ::Float64,
        times::Vector{Float64}, S::Vector{Int}, I::Vector{Int}, R::Vector{Int},
		Q, t_max::Float64, nodes::Vector{Node})

	#println("typeof(Glist)= ", typeof(Glist))
    if (S[end] <= 0)
		println("S=$(S[end]), (ERROR!!! S cannot be zero at this point")
	end
	node_u.status = :I
    push!(times, t)
    push!(S, S[end]-1)
    push!(I, I[end]+1)
    push!(R, R[end])
	# rec_time: time at which infected person recovers
	node_u.rec_time = t + rand(Exponential(γ))

	if node_u.rec_time < t_max
		new_event = Event(node_u, node_u.rec_time, :recover)
		Q[new_event] = new_event.time
	end

	w::Float64 = 0.

	status = getStatus(t, p_status[node_u.index])
	#println("typeof(Glist)= ", typeof(Glist))
	G = Glist[status]
	wghts = weights(G)
	#@show G, node_u.index
	for ν in neighbors(G, node_u.index)
		w = wghts[node_u.index, ν]
		findTransSIR(Q, t, τ, w, node_u, nodes[ν], t_max, nodes)
	end
end


function findTransSIR(Q, t, τ, w, source, target, t_max, nodes)
    # w is the edge weight
	if target.status == :S
		inf_time = t + rand(Exponential(τ*w))
		#print("inf_time")
		# Allocate memory for this list
		if inf_time < minimum([source.rec_time, target.pred_inf_time, t_max])
			new_event = Event(target, inf_time, :transmit)
			Q[new_event] = new_event.time
			target.pred_inf_time = inf_time
		end
	end
end

function processRecSIR(node_u, t, times, S, I, R)
	push!(times, t)
	push!(S, S[end])
	push!(I, I[end]-1)
	push!(R, R[end]+1)
	node_u.status = :R
end

#
# Notice the type of G is Any, meaning that it will accept a number,
# but then will crash inside the program. The type should probably be
# AbstractGraph, but then, it will have a problem with SimpleWeighteGraphs
function simulate(G, params, infected)
	#global γ, τ
	println("simulate, infected: $(typeof(infected))")
	times, S, I, R = fastSIR(G, params, infected)
	return times, S, I, R
end


end
