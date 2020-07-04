using LightGraphs, MetaGraphs
using GraphRecipes
import Plots
using CSV
using DataFrames, DataFramesMeta
using Makie

# I am having problems with backends. I need interactivity.

const all_df = CSV.read("all_data.csv")
const df_s = @where(all_df, :school_id .!== -1)
const df_w = @where(all_df, :work_id .!== -1)
const df_h = all_df

# Retrieve all businesses with nb_employees ∈ [lb, ub]
function find(df::DataFrame, group_col::Symbol, selector::Symbol, lb::Int, ub::Int)
    grps = groupby(df, group_col)
    dd = combine(grps, :work_id => length => selector)
    dd = dd[lb .< dd[selector] .< ub, :]
    return dd
end

# Starting with the full database, first identify a
#    workplace with lb < nb_employees < ub
# Then identify all the employees' homes
# Return a MetaGraph with a list of all the homes with an employee in the
#    chosen workplace. The first node is the workplace, the remaining
#    nodes are the homes. The edges have been computed.
function randomBusinessWorkToHomes(all_df, df_w, lb, ub)
    w_count_df = find(all_df, :work_id, :counts, lb, ub)

    # Choose a random business a
    row = rand(1:nrow(w_count_df), 1)
    # work id of random business
    wid  = w_count_df[row,:].work_id[1]
    # Full dataframe for wid
    # List of people in workplace wid

    # The where macro works inside the function.
    wid_df = @where(df_w, :work_id .== wid)
    #wid_df = df_w[df_w.work_id .== wid]

    wlong = wid_df.wlong
    wlat = wid_df.wlat
    # Coordinates of people's homes who work in that business
    hlong = wid_df.hlong
    hlat = wid_df.hlat
    nb_edges = length(hlat)
    nb_nodes = nb_edges + 1

    # Create a graph that connects the business with
    # Node 1 is the workplace
    # Nodes 2:nb_nodes+1 are the employee homes
    graph = SimpleGraph(nb_nodes, 0)
    for i in 1:nb_edges
        add_edge!(graph, 1, i+1)
    end

    # create node coordinates
    xx = vcat(wlong[1], hlong)
    yy = vcat(wlat[1], hlat)
    graph = MetaGraph(graph)
    set_prop!(graph, :node_coords, [xx, yy])
    return graph, wid_df
end
# ---------------------------------------------------------------------
# Functions for
function collectGraphEdges(graph::MetaGraph; hplane=0.0, wplane=0.0)
    println("collectGraphEdges, graph: $graph")
    n = nv(graph)
    nes = ne(graph)
    xn = rand(n)
    yn = rand(n)
    zn = fill(hplane, n)
    zn[1] = hplane
    xe = zeros(Float32, 2*nes)
    ye = zeros(Float32, 2*nes)
    ze = zeros(Float32, 2*nes)
    for (i,e) in enumerate(edges(graph))
        xe[2*i-1] = xn[1]  # workplace
        xe[2*i]   = xn[dst(e)]
        ye[2*i-1] = yn[1]
        ye[2*i]   = yn[dst(e)]
        ze[2*i-1] = wplane
        ze[2*i]   = hplane
    end
    mgraph = MetaGraph(graph)
    set_prop!(mgraph, :ncoords, [xn, yn, zn])
    # xe[2i-1), xe[2i] are the x coordinates of one graph edge
    set_prop!(mgraph, :line_segments, [xe, ye, ze])
    return mgraph
end

# Randomly connect two graphs
# Choose nb_edges between the two graphs
function connectGraphs(g1::MetaGraph, g2::MetaGraph, nb_edges::Int)
    n1 = nv(g1)
    n2 = nv(g2)
    x1,y1,z1 = get_prop(g1, :ncoords)  # 3-vec
    x2,y2,z2 = get_prop(g2, :ncoords)
    ix1 = rand(1:ne(g1), nb_edges)
    ix2 = rand(1:ne(g2), nb_edges)
    xe = zeros(Float32, 2*nb_edges)
    ye = zeros(Float32, 2*nb_edges)
    ze = zeros(Float32, 2*nb_edges)
    for i in 1:nb_edges
        xe[2i-1] = x1[ix1[i]]
        xe[2i]   = x2[ix2[i]]
        ye[2i-1] = y1[ix1[i]]
        ye[2i]   = y2[ix2[i]]
        ze[2i-1] = z1[ix1[i]]
        ze[2i]   = z2[ix2[i]]
    end
    return xe, ye, ze
end

# Starting from a list of homes, identify all the workplaces
# of all the people living in all the homes
# Remove the input workplace from the list
function getHomes(wid_df; hplane=0., wplane=1.)
    # Starting from the homes, Identify all the workplaces.
    # identify the home coordinates of all the workers
    xh = wid_df.hlat;   # House coordinates
    yh = wid_df.hlong;
    h_df = DataFrame([xh,yh],[:hlat, :hlong])  # all work_id are unique
    # Remove all the rows whose wid is the business id in wid_df
    people_in_homes_df = innerjoin(h_df, all_df, on = [:hlat, :hlong])
    wlong, wlat = people_in_homes_df.wlong, people_in_homes_df.wlat
    slong, slat = people_in_homes_df.slong, people_in_homes_df.slat

    println("wlat people: ", wlat .-30.)
    println("hlat people: ", people_in_homes_df.hlat .-30.)
    return

    # Remove all the rows with :work_id == id
    id = wid_df.work_id[1]
    people_in_homes_df = antijoin(people_in_homes_df, wid_df, on = :work_id)

    # rows of h_df that have schools
    # rows of h_df that have workplaces
    pih = people_in_homes_df;
    school_df = pih[pih.school_id .!= -1,:];
    work_df = pih[pih.work_id .!= -1,:];

    # Remove from school_df all rows where the school is outside Leon County
    school_df = @where(school_df, (-1. .< :slat  .- 30. .< 1.) .&
                                  (-1. .< :slong .+ 84. .< 1.))
    # Remove from work_df all rows where the workplace is outside Leon County
    work_df = @where(work_df, (-1. .< :wlat  .- 30. .< 1.) .&
                              (-1. .< :wlong .+ 84. .< 1.))

    hslat  = school_df.hlat  .- 30.;
    hslong = school_df.hlong .+ 84.;
    slat   = school_df.slat  .- 30.;
    slong  = school_df.slong .+ 84.;

    println("==> Why is hwlat and hwlong missing?")
    println("==> work_df= $work_df")
    #return
    hwlat  = work_df.hlat    .- 30.;
    hwlong = work_df.hlong   .+ 84.;
    wlat   = work_df.wlat    .- 30.;
    wlong  = work_df.wlong   .+ 84;

    # Create two graphs, one for schools, one for homes
    # These two graphs connect the home network to the school/work network
    graph_s = SimpleGraph(nrow(school_df)*2)
    graph_w = SimpleGraph(nrow(work_df)*2)
    println("nb nodes(graph_s): ", nv(graph_s))
    println("nb nodes(graph_w): ", nv(graph_w))

    # edges are 1-1 links between homes and schools
    nb_edges_s = nrow(school_df)
    for i in 1:nb_edges_s
        add_edge!(graph_s, i, i+nb_edges_s)
    end

    # edges are 1-1 links between homes and workplaces
    nb_edges_w = nrow(work_df)
    for i in 1:nb_edges_w
        add_edge!(graph_w, i, i+nb_edges_w)
    end

    # Construct a graph that connects schools and homes
    println("graph_w: ", graph_s)
    println("nb_edges_w: ", nb_edges_s)
    println("hslat: $(length(hslat))")
    println("hslong: $(length(hslong))")
    println("hwlat: $(length(hwlat))")
    println("hwlong: $(length(hwlong))")
    println("slat: $(length(slat))")
    println("slong: $(length(slong))")

    nb_edges_w = nrow(work_df)
    xxs = vcat(hslat, slat)
    yys = vcat(hslong, slong)
    zzs = fill(wplane, 2*nb_edges_s)  # school and workplace on the same plane
    zzs[1:nb_edges_s] .= hplane
    graph_s = MetaGraph(graph_s)
    set_prop!(graph_s, :node_coords, [xxs,yys,zzs])

    # Construct a graph that connects workplace and homes
    nb_edges_s = nrow(school_df)
    println("hwlat= ", hwlat)    # missing
    println("hwlong= ", hwlong)  # missing
    println("hslat= ", hslat)
    println("hslong= ", hslong)
    println("wlat= ", wlat)
    println("wlong= ", wlong)
    println("slat= ", slat)
    println("slong= ", slong)
    xxw = vcat(hwlat, wlat)
    yyw = vcat(hwlong, wlong)
    zzw = fill(wplane, 2*nb_edges_w)
    zzw[1:nb_edges_w] .= hplane
    graph_w = MetaGraph(graph_w)
    set_prop!(graph_w, :node_coords, [xxw,yyw,zzw])

    return [graph_s, graph_w]
end

function setupCamera!(scene)
    cam = cam3d!(scene)
    #cam.projectiontype[] = AbstractPlotting.Orthographic
    #cam.upvector = Vec3f0(0,0,1)
    eyepos = Vec3f0(5, 1.5, 0.5);
    lookat = Vec3f0(0);
    update_cam!(scene, eyepos, lookat);
end
# ----------------------------------------------
# Remove while debugging
#graph, wid_df = randomBusinessWorkToHomes(all_df, df_w, 3, 5)
xx, yy = get_prop(graph, :node_coords)
# the first node is the workplace, the others, the home
hplane = 0.7
wplane = 0.25
mgraph = collectGraphEdges(graph, hplane=hplane, wplane=wplane)
graph_s, graph_w = getHomes(wid_df, hplane=hplane, wplane=wplane);

# Could run in a separate thread
# Empty scene takes a long time to create. Weird.
scene = Scene()
x,y,z = get_prop(mgraph, :line_segments)
linesegments!(scene, x,y,z, color=:blue)
x,y,z = get_prop(mgraph, :ncoords)
scatter!(scene, x,y,z, markersize=.02, color=:blue)
setupCamera!(scene)

xs, ys, zs = get_prop(graph_s, :node_coords);
xw, yw, zw = get_prop(graph_w, :node_coords);
linesegments!(scene, xs, ys, zs, color=:green)
# MISSING VALUES in work arrays xw, yw
linesegments!(scene, xw, yw, zw, color=:orange)

# wid_df: list of homes, all different (not guaranteed)
# graph_s: list of schools connected to the homes
# graph_w: list of workplaces connected to the homes
# by connected: at least one member of each
function connectGraphs(wid_df, graph_s, graph_w)

# ----------------------------------------------------------------------
function drawGraph!(scene, mgraph::MetaGraph; edge_color=:black)
    xn, yn, zn = get_prop(mgraph, :ncoords)
    xe, ye, ze = get_prop(mgraph, :ecoords)
    scatter!(scene, xn, yn, zn, color=edge_color, markersize=.02)
    linesegments!(scene, xe, ye, ze, color=edge_color)
end


# wid_df is DataFrame of businesses with a number of employees within a specified range
graph, wid_df = randomBusinessWorkToHomes(all_df, df_w, 50, 200)
xx, yy = get_prop(graph, :node_coords)
#graphplot(graph, curvature=.01, nodesize=.05, x=xx, y=yy)






# ----------------------------------------------------------------------
# Plot these graph with Makie
Plots.display(p)
PyPlot.ion()
Plots.plotly()
PyPlot.display(p)

plt = Plots
function plotPatch()
    fig = PyPlot.figure()
    ax = PyPlot.Axes3D(fig)
    ax[:mouse_init]()
    ax[:set_xlim3d](left=-1.0, right=1.0)
    ax[:set_ylim3d](bottom=-1.0, top=1.0)
    ax[:set_zlim3d](bottom=-1.5, top=1.5)
end

plotPatch()
p = getHomes(wid_df)
PyPlot.display(p)
