# Based off the latest code written by Bryan, with a network closer to demographics
include("./modules.jl")
using Plots
include("./functions.jl")

# 0.65, skipping over sizes > 1000
# Questions to Answer:
# 1) Why do schools have sz > 5000? In fact, why are sizes > 100,000?
#    And this is after I removed the missing data from the dataframes
@time home_graph, work_graph, school_graph = generateDemographicGraphs(p)

@show school_graph
@show work_graph
@show home_graph

Glist = [home_graph, work_graph, school_graph]
@show total_edges = sum(ne.(Glist))

# Test this on simpler graphs with all to all connections
# Make the three graphs the same
function setupTestGraph(nv, ne)
    home_graph = SimpleGraph(nv, ne) #div(n*(n-1),2))
    work_graph =  deepcopy(home_graph)
    school_graph =  deepcopy(home_graph)
    Glist = [home_graph, work_graph, school_graph]
    Glist = MetaGraph.(Glist)
    for i in 1:3
        set_prop!(Glist[i], :node_ids, collect(1:n))
    end
    return Glist
end

#n = 300000
#Glist = setupTestGraph(n, 1000000)
#w = weights.(Glist);

# Retrieve all businesses with nb_employees ∈ [lb, ub]
function findInfected(df::DataFrame, group_col::Symbol, target_col::Symbol, lb::Int, ub::Int)
    grps = groupby(df, group_col)
    dd = combine(grps, group_col => length => target_col)
    dd = dd[lb .< dd[target_col] .< ub, :]
    ix = rand(dd[:,group_col], 1)[1]
    infected = df[df[group_col] .== ix,:]
    infected = infected.index
    return dd, ix, infected
end

dfs_work, ix, infected_0 = findInfected(all_df, :work_id, :count, 100, 200)

# *****************
# Using BSON, time to save is tiny fraction of total time
@time times, S,I,R = FPLEX.fastSIR(Glist, p, infected_0;
    t_max=1000., save_data=true, save_freq=5);
@show maximum.([S,I,R])
@show minimum.([S,I,R])

F.myPlot(times, S, I, R)
@show pwd()

nothing
# TODO:
# Check the weights on the three graph edges.
# Assign weights to graphs taken from P(w; mean, sigma)
# Run the SIR model
# Model will be α . β . γ . δ
# where α is the contact probability, β is the infectiousness,
# γ is the efficacy of masks and δ is the co-morbidity factor
# These four parameters will have their own distributions.
# Therefore, we can infer their parameters from various inference experiments.
# For that, we may have to read papers on inference. That is for later.

@time data = rand(1000, 1000)
@time F.heatmap(data)

#Take a function:

function f(x, y)
    exp(-x^2 + y)
end

x = randn(2000000)
y = randn(2000000)
fct = f.(x, y)
@time histogram2d(x, y, nbins=150)

function readData()
    filename = "Leon_Formatted/people_formatted.csv"
    populace_df = loadPickledPop(filename)
    rename!(populace_df, :Column1 => :person_id)

    # replace 'X' by -1 so that columns are of the same type
    df = copy(populace_df)
    replace!(df.work_id, "X" => "-1")
    replace!(df.school_id, "X" => "-1")

    tryparsem(Int64, str) = something(tryparse(Int64,str), missing)
    df.school_id = tryparsem.(Int64, df.school_id)
    df.work_id   = tryparsem.(Int64, df.work_id)
    return df
end

df = readData()
filename = "Leon_Formatted/households_formatted.csv"
dfh = CSV.read(filename, delim=',')
lat = dfh.latitude .- 30.
long = dfh.longitude .+ 84.

filename = "Leon_Formatted/workplaces_formatted.csv"
dfw = CSV.read(filename, delim=',')
latw = dfw.latitude .- 30.
longw = dfw.longitude .+ 84.

filename = "Leon_Formatted/schools_formatted.csv"
dfs = CSV.read(filename, delim=',')
lats = dfs.latitude .- 30.
longs = dfs.longitude .+ 84.

#@time histogram2d(long, lat, nbins=100)
@time ss = scatter(long, lat, xlabel="long", ylabel="lat",
    markersize=.05, aspect_ratio=1, label="")

xs = xlims(ss)
ys = ylims(ss)
@time scatter!(longs, lats, xlabel="long", ylabel="lat",
    markersize=4, aspect_ratio=1, label="", alpha=.5,
    xlims=xs, ylims=ys)

@time scatter!(longw, latw, xlabel="long", ylabel="lat",
    markersize=1, aspect_ratio=1, label="", alpha=.5,
    xlims=xs, ylims=ys)

# For each person, I need lat/long for workplace/school and for home.
# lath/longh and latw/longw (which includes school)

# Merge dfh and df
# Merge dfs and df
# perhaps join along
#dfh.sp_id  and   df.sp_hh_id


dfh.sp_hh_id = dfh.sp_id
