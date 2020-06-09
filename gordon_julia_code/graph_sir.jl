using Plots
using DataFrames
using ModelingToolkit
using DiffEqBase
using DiffEqJump
using LightGraphs
using SpecialGraphs
using DiffEqBase.EnsembleAnalysis
using Profile
#using PProf

const nedgess = 300
const nvertss = 500

#const dgs = DiGraph(nvertss, nedgess);
# Degree of each node: 2nd argument  5
const dgr = random_regular_digraph(nvertss, 5)

function setupSystem(graph)
    nverts = length(vertices(graph))
    nedges = length(edges(graph))
    # Allow for different beta on each edgse
    @parameters t β[1:nedges]  γ[1:nverts] ;
    @variables S[1:nverts](t);
    @variables I[1:nverts](t);
    @variables R[1:nverts](t);

    rxsS = [Reaction(β[i],[I[src(e)], S[dst(e)]], [I[dst(e)]], [1,1], [2])
        for (i,e) ∈ enumerate(edges(graph))]

    rxsI = [Reaction(γ[v],[I[v]], [R[v]])  # note: src to src, yet there is no edges
        for v ∈ vertices(graph)]

    rxs = vcat(rxsS, rxsI);
    vars = vcat(S,I,R);
    params = vcat(β,γ);

    @time rs = ReactionSystem(rxs, t, vars, params);
    println("++++ Completed convert to ReactionSystem\n")

    @time js = convert(JumpSystem, rs);
    println("++++ Completed convert to JumpSystem\n")

    S0 = ones(nverts)
    I0 = zeros(nverts)
    R0 = zeros(nverts)

    function infect(i)
        S0[i] = 0; # One person is infected
        I0[i] = 1;
        R0[i] = 1 - S0[1] - I0[1]
    end

    #infect.([1,10,15,25])
    infect.([3]) #,10,15,25])
    vars0 = vcat(S0, I0, R0);

    # Two column vectors
    γ = fill(0.25, nverts);
    β = fill(0.50, nedges);
    params = vcat(β,γ)

    initial_state = [convert(Variable,state) => vars0[i] for (i,state) in enumerate(states(js))];
    initial_params = [convert(Variable,par) => params[i] for (i,par) in enumerate(parameters(js))];

    tspan = (0.0,20.0)
    t_dense = 0.0,:.15:tspan[2]

    #println("*** typeof(js): ", typeof(js))
    #println("*** which: ", @which DiscreteProblem(js, initial_state, tspan, initial_params))
    #println("*** llvm: ", @code_llvm DiscreteProblem(js, initial_state, tspan, initial_params))
    #dprob = @profile DiscreteProblem(js, initial_state, tspan, initial_params)
    @time dprob = DiscreteProblem(js, initial_state, tspan, initial_params)
    println("++++ Completed DiscreteProblem\n")
    println()

    @time jprob = JumpProblem(js, dprob, NRM(); save_positions=(false, false))
    println("++++ Completed JumpProblem\n")

    # Save solution at times 0., 0.25, 0.50, etc
    #@time sol = solve(jprob, SSAStepper(), saveat=0.25)
    #println("Completed: solve")

    return jprob
end


function processData(sol)
    nverts = nvertss
    nedges = nedgess
    println("nverts=$nverts, nedges= $nedges")

    dfs = convert(Matrix, DataFrame(sol))
    Sf = dfs[1:nverts,:]
    If = dfs[nverts+1:2*nverts,:]
    Rf = dfs[2*nverts+1:3*nverts,:]
    Savg = (sum(Sf; dims=1)') / nverts
    Iavg = (sum(If; dims=1)') / nverts
    Ravg = (sum(Rf; dims=1)') / nverts
    print(Savg)
    return Savg, Iavg, Ravg
end

# Times: sol.t
# Solution at nodes: sol.u
# sol.u[1] |> length == 120 (3 * nverts)
@profile jprob = setupSystem(dgr);
Profile.print()
#Juno.profiler()

# Solve the solution nb_iter times, and compute averaged quantities
# EnsembleProblem does not help because 1) there is no support for graphs,
# 2) I have (SIR) equations at each graph vertex, and I must disambiguate
# them manually. Since I want standard deviations, at each time step,
# I will average all the nodes (allowed since they all have the same degree)
# Given a solution s(t_i), I could trick EnsembleProblems by solving the
# ODE that would give s(t) as a solution. This would be a discrete problem
# with  s[i+1] = s(t_{i+1}). Alternatively, I can do the calculations by hand.

# sol0[time], where each sol0[i] is a solution over the graph

# sol.u[1,:], sol.u[2,:],...,are row vectors at fixed time.
# Stack them vertically
# More information: https://stackoverflow.com/questions/24522638/julia-transforming-an-array-of-arrays-in-a-2-dimensional-array
# I do not understand why hcat works

function multiSamples(jprob; nb_runs=5)
    println("\n*** Perform $nb_runs sample runs ***")
    sol0 = solve(jprob, SSAStepper(), saveat=0.50);
    t = sol0.t
    mean = hcat(sol0.u...); # Creates a 2D matrix (nodes, time)
    var = mean .^ 2

    for i in 2:nb_runs
        sol0 = hcat(solve(jprob, SSAStepper(), saveat=0.50).u...)
        mean .+= sol0
        var .+= sol0.^2
    end

    mean ./= nb_runs
    var .= (var ./ nb_runs .- mean.^2)
    return mean, var, t
end;
3+3
function processSolution(solm, solvar)
    nverts, ntimes = size(solm)
    nvars = 3   # number variables: S, I, R
    nverts = div(nverts, nvars) # integer division
    ix = nverts
    Sf = solm[1:ix, :]
    If = solm[ix+1:2*ix, :]
    Rf = solm[2*ix+1:3*ix, :]
    Savg = (sum(Sf; dims=1)') / nverts
    Iavg = (sum(If; dims=1)') / nverts
    Ravg = (sum(Rf; dims=1)') / nverts

    Sv = solvar[1:ix,:]
    Iv = solvar[ix+1:2*ix, :]
    Rv = solvar[2*ix+1:3*ix, :]
    Sstd = sqrt.((sum(Sv; dims=1)') / nverts)
    Istd = sqrt.((sum(Iv; dims=1)') / nverts)
    Rstd = sqrt.((sum(Rv; dims=1)') / nverts)
    return Savg, Iavg, Ravg, Sstd, Istd, Rstd
end

@time mean, var, t = multiSamples(jprob, nb_runs=20)

@time Savg, Iavg, Ravg, Sstd, Istd, Rstd =
     processSolution(mean, var);

plot(t, Savg, grid=false, ribbons=Sstd, fillalpha=0.2)
plot!(t, Iavg, grid=false, ribbons=Istd, fillalpha=0.2)
plot!(t, Ravg, grid=false, ribbons=Rstd, fillalpha=0.2)

3
