
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
    Event(node::Node, time::Float64, action::Symbol) =
        new(node, time, action)
end

function fastSIR(G, params, initial_infecteds::Vector{Int})
	τ = params.τ
	γ = params.γ
	expo_γ = Exponential(γ) # type Distribution::Exp
	expo_τ = Exponential(τ)
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
                @time processTransSIR(G, event.node, event.time, τ, γ, times,
					S, I, R, Q, t_max, nodes, expo_τ, expo_γ)
            end
		else
             # 1 alloc: 16 bytes, 0.000001 to 0.000002 seconds
             processRecSIR(event.node, event.time, times, S, I, R)
        end
    end
	println("times, times[end]: $(length(times)), $(times[end])")
    times, S, I, R
end;

function processTransSIR(G, node_u, t::Float64, τ::Float64, γ::Float64,
        times::Vector{Float64}, S::Vector{Int}, I::Vector{Int}, R::Vector{Int},
		Q, t_max::Float64, nodes::Vector{Node}, expo_τ, expo_γ)
    if (S[end] <= 0)
		println("S=$(S[end]), (ERROR!!! S cannot be zero at this point")
	end
	node_u.status = :I
    push!(times, t)
    push!(S, S[end]-1)
    push!(I, I[end]+1)
    push!(R, R[end])
	#node_u.rec_time = t + rand(Exponential(γ))
	node_u.rec_time = t + rand(expo_γ)

	if node_u.rec_time < t_max
		@time new_event = Event(node_u, node_u.rec_time, :recover)
		println("after Event\n")
		@time Q[new_event] = new_event.time
		println("after Q\n")
	end
	@time for ν in neighbors(G, node_u.index)
		@time findTransSIR(Q, t, τ, node_u, nodes[ν], t_max, nodes, expo_τ)
		println("after findTransSIR (within neighbor loop)\n")
	end
	println("after neighbors\n")
end


function findTransSIR(Q, t, τ, source, target, t_max, nodes, expo_τ)
	if target.status == :S
		#inf_time = t + rand(Exponential(τ))
		inf_time = t + rand(expo_τ)
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

function simulate(G, params, infected)
	#global γ, τ
	times, S, I, R = fastSIR(G, params, infected)
	return times, S, I, R
end

function myPlot(times, S, I, R)
   plot(times, S, label=:S)
   plot!(times, I, label=:I)
   plot!(times, R, label=:R)
end
