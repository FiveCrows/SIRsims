module FastSIRWithWeightedNodes

using LightGraphs
import Random: seed!
using Distributions
using DataStructures

# Consumes too much memory
mutable struct Node
    index::Int  # :S, :I, :R
    status::Symbol  # :S, :I, :R
    pred_inf_time::Float64
    rec_time::Float64
    #Node(index::Int, status::Symbol, pred_inf_time::Float64, rec_time::Float64) =
        #new(index, status, pred_inf_time, rec_time)
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

function fastSIR(G, params, initial_infecteds::Vector{Int})
	τ = params.τ
	γ = params.γ
	# These can no longer be precomputed because in principle, each
	# edge can have a different weight.
	t_max = params.t_max  # mandatory

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

    for u in 1:nb_nodes  # extremely fast
        nodes[u] = Node(u, :S, pred_inf_time, rec_time)
    end

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
                # 12 allocations, 224 bytes
                processTransSIR(G, event.node, event.time, τ, γ, times,
					S, I, R, Q, t_max, nodes)
                #println("processTransSIR\n")
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


function processTransSIR(G, node_u, t::Float64, τ::Float64, γ::Float64,
        times::Vector{Float64}, S::Vector{Int}, I::Vector{Int}, R::Vector{Int},
		Q, t_max::Float64, nodes::Vector{Node})
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

	wghts = weights(G)
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
