# Duplicate results of graph-based SIR with ReactionSystem in ModelingToolkit
# with Gillepsie. The machinery seems lighter.

using Plots
using DataFrames
using LightGraphs
using Gillespie
using Profile
import Random: seed!
# for jjensen
using Distributions
using StaticArrays

const nvertss = 5

#const dgs = DiGraph(nvertss, nedgess);
# Degree of each node: 2nd argument  5
#const dgr = random_regular_digraph(nvertss, 2)
dgrsss = random_regular_digraph(nvertss, 2)

#----------------------------
function jjensen(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64,max_rate::Float64,thin::Bool=true)
    if thin==false
      return jensen_alljumps(x0::AbstractVector{Int64},F::Base.Callable,nu::Matrix{Int64},parms::AbstractVector{Float64},tf::Float64,max_rate::Float64)
    end

    # I need an inplace reshape
    x0 = reshape(x0, nvertss, 3); # experimental

    tvc=true
    try
      F(x0,parms,0.0)
    catch
      tvc=false
    end

    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:jensen,tvc)
    #println("GE: args: ", args)

    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')  # orig  (why transpose?)
    x = copy(x0)  #  Each row is one solution at a node
    xa = [copy(x0)  # what is xa?
    # Number of propensity functions; one for no event
    numpf = size(nu,1)+1
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    count = 0
    while t <= tf
        dt = rand(Exponential(1/max_rate))
        t += dt
        println("dt= $dt, t=  $t, tf= $tf")
        if tvc
          pf = F(x,parms,t)
        else
           #println("tvc= ", tvc)  # tvc = false
          pf = F(x,parms)  # rates: infection, recovery
          println("x= ", x)
          println("pf= ", pf)
        end
        # Update time (0.3 continuous and then 32 then 101.3. WHY?)
        #println("pf= ", typeof(pf), size(pf))
        #println("pf= ", pf)
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        #println("sumpf= ", sumpf)
        if sumpf > max_rate
            termination_status = "upper_bound_exceeded"
            break
        end
        # Update event
        println("numpf+1= ", numpf+1)
        println("[pf;max_rate-sumpf]= ", [pf; max_rate-sumpf])
        ev = pfsample([pf; max_rate-sumpf],max_rate,numpf+1)
        println("numpf= ", numpf, ",   ev= ", ev)
        if ev < numpf
            if x isa SVector
                #println("x isa SVector")
                @inbounds x[1] += nu[ev,:]
            else
                #println("x is NOT SVector")
                println("xbefore= ", x)
                #println("nu=  $(size(nu)),  $nu")
                #println("ev= ", ev)
                #println("view(nu,ev) = ", view(nu,ev))
                #println("view(nu,ev,:) = ", view(nu,ev,:))
                deltax = view(nu,ev,:)
                #for i in 1:nstates  # Probably an error. Only one vertex updated at a time
                for i in 1:3   # 3 is number of states per node
                    #println("deltax[$i] = ", deltax[i])
                    #println("size(x)= ", size(x))  # 1 x 15
                    #println("size(x0)= ", size(x0))  # 1 x 15
                    #@inbounds x[1,i] += deltax[i]
                    @inbounds x[ev,i] += deltax[i]
                end
                println("xafter= ", x)
            end
					count = count + 1
					#=
          for xx in x
            println("--------------------------, count= $count")
            #println("xx=  $(typeof(xx)), $(size(xx)), $xx")
            println("xa=  $(typeof(xa)), $(size(xa)), $xa")
            println("t= $t, dt= $dt")
            push!(xa,xx)
          end
					=#
          push!(ta,t)
          # update nsteps
          nsteps += 1
        end
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
	#print(typeof(xar))
	#print(typeof(stats), stats)
    return SSAResult(ta,xar,stats,args)
end
result = jjensen(vars0, F_sir, nu,parms, tf, 100.)

#----------------------------

# create a system with all S, I, R equations
# I think that systems are more complicated
# Consider two vertices: v1, v2 and one edge: e
# S1 + I2 --> I1 + I2  [-1 1 0]
# S2 + I1 --> I1 + I2  [-1 1 0]   # (if bidirectional link)
# I1      --> R1  [0 -1 1]
# I2      --> R2  [0 -1 1]
function F_sir(x,parms)
	S = @view x[0*nvertss+1:1*nvertss]
	I = @view x[1*nvertss+1:2*nvertss]
	R = @view x[2*nvertss+1:3*nvertss]
    (beta,gamma) = parms
	#print("edges(dgrsss)= ", collect(edges(dgrsss)))
	#println("F_SIR, x= ", size(x))
	#println("F_SIR, S= ", S)
	#println("F_SIR, I= ", I)
	#println("F_SIR, R= ", R)
	infections = [beta*S[src(e)]*I[dst(e)]
			for e ∈ edges(dgrsss)]
	recoveries = [gamma*I[v]
			for v ∈ vertices(dgrsss)]
				#println("infections= ", infections)
				#println("recoveries= ", recoveries)
    vcat(infections, recoveries)
end

nu1a= repeat([-1 1 0], nvertss)
nu1b = repeat([0 -1 1], nvertss)
nu = vcat(nu1a, nu1b)
tf = 200.0
nverts = nvertss

S0 = ones(Int, nverts)
I0 = zeros(Int, nverts)
R0 = zeros(Int, nverts)

function infect(i)
    S0[i] = 0; # One person is infected
    I0[i] = 1;
    R0[i] = 1 - S0[1] - I0[1]
end

#infect.([1,10,15,25])
infect.([3]) #,10,15,25])
vars0 = vcat(S0, I0, R0);
println("size(vars0)= ", size(vars0))
seed!(1234);

parms = [0.1, 0.1];
#result = ssa(vars0, F_sir, nu,parms, tf)
result = jjensen(vars0, F_sir, nu,parms, tf, 100.);;
data  = ssa_data(result);
print(first(data, 10));
