# Based off the latest code written by Bryan, with a network closer to demographics
using Plots
default(legendfontsize=10)
include("./pickle.jl")
include("./modules.jl")
# Generates and sets up the graphs
include("./functions.jl")
include("./read_age_map.jl")
include("./make_contact_graph_functions.jl")

# 0.65, skipping over sizes > 1000
# Questions to Answer:
# 1) Why do schools have sz > 5000? In fact, why are sizes > 100,000?
#    And this is after I removed the missing data from the dataframes

const all_df = CSV.read("all_data.csv")
# Replace all missing by -1
# Assume that all missings are integers (I do not know that to be true)
all_df = coalesce.(all_df, -1)
all_df.index = collect(1:nrow(all_df))
person_ids = all_df[:person_id] |> collect
# Add to all_df, the number of people in each workplace
work_groups = groupby(all_df, "work_id")
all_df = transform(work_groups, nrow => :wk_count)

# make the minimum age 1 to be consistent with the Transmission Matrix data
replace!(all_df.age, 0 => 1)


# deg_graph_dict[school_id] = (degrees, N_tot, N_age[i], index_range)
# Ntot = sum(N_age)
@time home_graph, work_graph, school_graph, deg_graph_dict = generateDemographicGraphs(p)
a = collect(deg_graph_dict)
println("a[1]= $(a[1])")

function plotDegGraph(deg_graph; stype="")
    # Values of dictionary: (degrees, N_tot, N_age[i], index_range)
    yy = collect(values(deg_graph)) |> deepcopy
    pp = plot(displaysize=(1200,800))
    for (i,z) in enumerate(yy)
        tup = yy[i]
        yy[i] = (tup[2], tup[1], tup[3], tup[4])
    end
    yy = sort(yy, rev=true)
    fntsm = Plots.font("sans-serif", pointsize=round(10.0*0.4))
    default(legendfont = fntsm)

    for (i,k) in enumerate(yy)
        N, dg, N_age, index_range = yy[i]
        if i > 20 break end
        label = string(N) * ", " * string(N_age)
        if i <=5 style = :solid;  shape=:square
        elseif i <=10 style = :solid; shape=:circle
        elseif i <=15 style = :solid;  shape=:dtriangle
        elseif i <=20 style = :solid;  shape=:utriangle end
        plot!(dg, label=label, fontsize=5, style=style, shape=shape,
            msize=3.0, msc=:match, msw=0, size=(800,600))
        xlabel!("Degree")
        ylabel!("Number of nodes")
        if stype == ""
            title!("Degree distributions in Schools")
        else
            title!("Degree distributions (" * stype * ")")
        end
    end
    return pp
end
pp = plotDegGraph(deg_graph_dict)

# Separate out elementary from high schools
s00xx_dict = Dict()
s0xxx_dict = Dict()
sxx00_dict = Dict()
sxxxx_dict = Dict()
for k in keys(deg_graph_dict)
    v = deg_graph_dict[k]
    N_age = v[3]

    if N_age[1] == 0 && N_age[2] == 0
        s00xx_dict[k] = v
    elseif N_age[1] == 0 && N_age[3] != 0
        s0xxx_dict[k] = v
    elseif N_age[3] == 0 && N_age[4] == 0
        sxx00_dict[k] = v
    else
        sxxxx_dict[k] = v
    end
end
p1 = plotDegGraph(s0xxx_dict, stype="s0xxx")
p2 = plotDegGraph(s00xx_dict, stype="s00xx")
p3 = plotDegGraph(sxx00_dict, stype="sxx00")
p4 = plotDegGraph(sxxxx_dict, stype="sxxxx")
pp = plot(p1,p2,p3,p4, displaysize=(2400,1800), layout=(2,2))
savefig(pp, "gordon.pdf")
plot(p1,p2,p3,p4, layout=(2,2))

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
# According to Holme, one should only start a simulation with one infected.
function setInfected(df::DataFrame, group_col::Symbol, target_col::Symbol, lb::Int, ub::Int)
    grps = groupby(df, group_col)
    dd = combine(grps, group_col => length => target_col)
    dd = dd[lb .< dd[target_col] .< ub, :]
    ix = rand(dd[:,group_col], 1)[1]
    infected = df[df[group_col] .== ix,:]
    infected = infected.index
    return dd, ix, infected
end

function setRecovered(df::DataFrame, group_col::Symbol, target_col::Symbol, lb::Int, ub::Int)
    grps = groupby(df, group_col)
    dd = combine(grps, group_col => length => target_col)
    dd = dd[lb .< dd[target_col] .< ub, :]
    ix = rand(dd[:,group_col], 1)[1]
    recovered = df[df[group_col] .== ix,:]
    recovered = recovered.index
    return dd, ix, recovered
end

# collect the indices of people whose target_col is witin [lb, ub]
# For example, getIndexes(df, :age, 8, 15) retrieves all the people whose age
# is between lb and ub
function getIndexes(df::DataFrame, target_col::Symbol, lb::Int, ub::Int)
    indexes = df[lb .<= df[target_col] .<= ub, :].index
    return indexes
end

recovered_0 = getIndexes(all_df, :age, 0, 18)
dfs_work, ix, infected_0 = setInfected(all_df, :work_id, :count, 100, 200)
#dfs_school, ix, recovered_0 = setRecovered(all_df, :school_id, :count, 400, 800)

# Prepare weights to take contact matrices into account.
# I have copied the transmission matrix from the paper by Valle
#transmission_map_file = "transmission_map_age_matrix.csv"
#df_map = CSV.read(transmisison_map_file, sep=',')

### Sets up ages between 1 in 90
### I should assume the gages go to 100
g = SimpleGraph(1000, 5000)
g = MetaGraph(g)
using LightGraphs, MetaGraphs
include("./read_age_map.jl")

# Graphs in Glist are MetaGraphs
# Add an edge dictionary to the graph as a global graph property.
function setupMetaData(all_df, Glist)
    for g in Glist
        setupTransmission(all_df, g)

        # create edge dictionary
        edge_dict = Dict()
        for e in collect(edges(g))
            # Necessary for undirected graph
            edge_dict[(src(e),dst(e))] = e
            edge_dict[(dst(e),src(e))] = e
        end
        set_prop!(g, :edge_dict, edge_dict)  # not the best approach
        # First get the code running
    end
end

setupMetaData(all_df, Glist)

include("./modules.jl")

# ****************************************************
# Using BSON, time to save is tiny fraction of total time
t_max = 500.   # Artificial limit! Remove when code runs.
@time times, S,I,R = FPLEX.fastSIR(Glist, p, infected_0, recovered_0;
    t_max=t_max, save_data=true, save_freq=10);
# ****************************************************
@show maximum.([S,I,R])
@show minimum.([S,I,R])



#--------------------------------------------------------------

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
