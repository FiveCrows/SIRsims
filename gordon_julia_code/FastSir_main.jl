
include("FastSIR_graphs.jl")
F = FastSIR

#using DataStructures
using LightGraphs
#using Distributions
#using Plots

# const G = F.erdos_renyi(200000, 0.0002)
# Erdos-renyi(200000, 0.0002) generates a graph with 200,000 edges and 4M edges.
# The cost of fastSIR in Jlia is 8.8 sec, 400,000 iterations, and 740M memory allocs
# So create a smaller graph and debug

# const G = F.erdos_renyi(50000, 0.0002)
# each call to processTransSIR: 0.000007 sec, 11 alloc, 210 bytes. WHY?

# t_max is mandatory parameter
const G = F.erdos_renyi(50000, 0.0002)
# Compute the maximum degree
@time max_degree = Δ(G)  # no allocations
@time neigh_list = zeros(Int, max_degree)
@time adj = adjacency_matrix(G)
println(adj[2])


println("Graph: $(nv(G)) nodes, $(ne(G)) edges")

#times, S, I = simulate()
# ρ: fraction initially infected
const params = (τ=.3, γ=1.0, t_max=5., ρ=0.05)

const infected = rand(1:nv(G), Int(floor(nv(G)*params.ρ)))
println("Initial number of infected: $(length(infected)),  percentage infected: $(params.ρ)")

# Higher τ means higher infection rate, so infection should grow faster. It does not.
# Higher τ means smaller time increments (smaller increment to infection, so infections should rise faster)

for i in 1:1
	@time global times, S, I, R = F.simulate(G, params, infected)
end
F.myPlot(times, S, I, R)

n = 39
tn = times[end-n:end];
Sn = S[end-n:end];
In = I[end-n:end];
Rn = R[end-n:end];
F.myPlot(tn, Sn, In, Rn);
F.myPlot(times, S, I, R);
#----------------------------------------------------------------------
include("parametrized_structs.jl")
parametrized_experiment()

include("non_parametrized_structs.jl")
non_parametrized_experiment()








# only 32 bytes allocated
@time for i in nv(G)
	neighbors(G, nodes[i].index)
end

#@time for n in nodes
count = [3]
count1 = [4]
# No memory allocation
@time for i in 1:nv(G) #(i,node) in enumerate(nodes)
	count[1] = count1[1] - count[1]
	count[1] = count1[1] + count[1]
	#neighbors(G, getIndex(n)) #getIndex(node))
	#print("gg")
end
print(count)
