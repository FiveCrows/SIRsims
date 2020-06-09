
include("FastSIR_graphs.jl")
F = FastSIR

using DataStructures
using LightGraphs
using Distributions
using Plots


function fastSIR(G, τ::Float64, γ::Float64,
           initial_infecteds::Vector{Int}, t_max::Float64)
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
    nodes = Array{F.Node,1}(undef, nb_nodes)

    for u in 1:nb_nodes  # extremely fast
        nodes[u] = F.Node(u, :S, pred_inf_time, rec_time)
    end

    for u in initial_infecteds
       #println("nodes[u]= ", nodes[u])
       event = F.Event(nodes[u], 0., :transmit)
       # How to do this with immutable structure
       nodes[u].pred_inf_time = 0.0  # should this new time be reflectd in the event? Here, it is.
       Q[event] = event.time
    end

    while !isempty(Q)
        event = dequeue!(Q)
        if event.action == :transmit
            if event.node.status == :S
                # 12 allocations, 224 bytes
                processTransSIR(G, event.node, event.time, τ, γ, times, S, I, R, Q, t_max, nodes)
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
        times::Vector{Float64}, S::Vector{Int}, I::Vector{Int}, R::Vector{Int}, Q, t_max::Float64, nodes::Vector{F.Node})
    push!(times, t)
    if (S[end] <= 0)
		println("S=$(S[end]), (ERROR!!! S cannot be zero at this point")
	end
    push!(S, S[end]-1)
    push!(I, I[end]+1)
    push!(R, R[end])
	node_u.rec_time = t + rand(Exponential(γ))
    #println("S= ", S)
    #println("I= ", I@time G = F.makeGraph(30000, 10)

	if node_u.rec_time < t_max
		new_event = F.Event(node_u, node_u.rec_time, :recover)
		Q[new_event] = new_event.time
	end
	for ν in neighbors(G, node_u.index)
		findTransSIR(Q, t, τ, node_u, nodes[ν], t_max, nodes)
	end
end

function findTransSIR(Q, t, τ, source, target, t_max, nodes)
	if target.status == :S
		inf_time = t + rand(Exponential(τ))
		# Allocate memory for this list
		if inf_time < minimum([source.rec_time, target.pred_inf_time, t_max])
			new_event = F.Event(target, inf_time, :transmit)
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

@time G = F.erdos_renyi(100000, 0.0008)
const GG = F.erdos_renyi(100000, 0.0008)

t_max = 20.
τ = 0.3
γ = 1.00

function simulate()
	global γ, τ
	infected = rand(1:nv(GG), div(nv(GG), 100))
	println("simulate: γ= $γ")
	times, S, I, R = fastSIR(GG, τ, γ, infected, t_max)
	return times, S, I, R
end

#times, S, I = simulate()
for i in 1:1
	global times, S, I, R = simulate()
end

nt = 500
plot(times[1:nt], [S[1:nt], I[1:nt], R[1:nt]])
# Susceptibles becoming negative!!
plot(times, S, label=:S)
plot!(times, I, label=:I)
plot!(times, R, label=:R)

a = zeros(10000)
b = 34.
c = ones(100000)
const a1 = zeros(10000)
const b1 = 34.
const c1 = ones(100000)
function tst(aa, bb, cc)
	return aa
end

function tst1(aa, bb, cc)
	for i in 1:1000
		push!(aa, 3.)
	end
	return aa
end

@time tst( a, b, c)
@time tst1(a1, b1, c1)
@time tst1(a, b, c)
