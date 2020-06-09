using ModelingToolkit
using DiffEqBase
using DiffEqJump
using LightGraphs
using SpecialGraphs

nedges = 7
nverts = 10

#dg = SpecialGraphs.CompleteGraph(nverts);
#nedges = ne(dg)

dg = DiGraph(nverts, nedges)
for i in 1:nverts
    add_edge!(dg, i, i)
end

@parameters t k[1:nedges];
@variables u[1:nverts](t);

#rxs = [Reaction(k[i],[u[src(e)]], [u[dst(e)]])
rxs = [Reaction(k[i],[u[src(e)]], [u[src(e)]])
    for (i,e) ∈ enumerate(edges(dg))]
rs = ReactionSystem(rxs, t, u, k)
js = convert(JumpSystem, rs)

# each vertex's value is a random integer in 1...100
u0map = [u[i] => rand(0:1) for i ∈ eachindex(u)]
pmap  = [k[i] => rand() for i ∈ eachindex(k)]
tspan = (0.0,10.0)
dprob = DiscreteProblem(js, u0map, tspan, pmap)
jprob = JumpProblem(js, dprob, NRM())
sol = solve(jprob, SSAStepper())



using Plots
plot(sol)
