#using Plots
using CSV
using DataFrames, DataFramesMeta
using Makie, GeometryTypes, AbstractPlotting
using LightGraphs, MetaGraphs

const AP = Makie

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
    # first node is the workplace,
    # remaining nodes are the homes
    xx = vcat(wlong[1], hlong)
    yy = vcat(wlat[1], hlat)
    graph = MetaGraph(graph)
    set_prop!(graph, :node_coords, [xx, yy])  # full long, lat
    set_prop!(graph, :h_coords, [hlong, hlat])  # same as xx, except one node
    return graph, wid_df
end
# ---------------------------------------------------------------------
# Functions for
function collectGraphEdges(graph::MetaGraph; hplane=0.0, wplane=0.0)
    n = nv(graph)
    nes = ne(graph)
    xn, yn = get_prop(graph, :node_coords)
    xn .+= 84.
    yn .-= 30.

    zn = fill(wplane, length(xn))
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
        if isa(graph, MetaGraph)
            mgraph = graph
        else
            mgraph = MetaGraph(graph)
        end
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

    h_df = wid_df[!, [:hlat, :hlong]]

    people_in_homes_df = innerjoin(h_df, all_df, on = [:hlat, :hlong])
    h_df = wid_df[!, [:work_id]]
    # keep rows from first arg that are not in 2nd arg
    people_in_homes_df = antijoin(people_in_homes_df, h_df, on = :work_id)
    println("nrow people_df: ", nrow(people_in_homes_df))

    wid_home_id = Set(wid_df.sp_hh_id)  # homes in wid_df are unique but do not have to be
    wid_hlat = Set(wid_df.hlat)  # homes in wid_df are unique but do not have to be
    wid_hlong = Set(wid_df.hlong)  # homes in wid_df are unique but do not have to be
    wid_work_id = Set(wid_df.work_id)  # homes in wid_df are unique but do not have to be

    # List of homes in people_in_homes_df
    pih = people_in_homes_df
    pih_home_id = Set(pih.sp_hh_id)
    pih_hlat = Set(pih.hlat)
    pih_hlong = Set(pih.hlong)
    pih_work_id = Set(pih.work_id)

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

    hwlat  = work_df.hlat    .- 30.;
    hwlong = work_df.hlong   .+ 84.;
    wlat   = work_df.wlat    .- 30.;
    wlong  = work_df.wlong   .+ 84;

    # Create two graphs, one for schools, one for homes
    # These two graphs connect the home network to the school/work network
    graph_s = SimpleGraph(nrow(school_df))
    graph_w = SimpleGraph(nrow(work_df))
    graph_s = MetaGraph(graph_s)
    graph_w = MetaGraph(graph_w)
    graph_s, graph_w = MetaGraph.(SimpleGraph.(nrow.([school_df, work_df])))

    # Construct a graph that connects schools and homes
    #=
    println("graph_w: ", graph_s)
    println("nb_edges_w: ", nb_edges_s)
    println("hslat: $(length(hslat))")
    println("hslong: $(length(hslong))")
    println("hwlat: $(length(hwlat))")
    println("hwlong: $(length(hwlong))")
    println("slat: $(length(slat))")
    println("slong: $(length(slong))")
    =#

    set_prop!(graph_s, :ncoords_s, [slong, slat,   fill(wplane, length(slong))])
    set_prop!(graph_s, :ncoords_h, [hslong, hslat, fill(hplane, length(hslong))])
    println("hslat graph_s: ", hslat)  # compare against homes in wid_df
    println("hslong graph_s: ", hslong)
    println("wid.home graph_s, lat: ", wid_df.hlat[1:5])
    println("wid.home graph_s, long: ", wid_df.hlong[1:5])
    println("wid.home graph_s, work_id: ", wid_df.work_id) # should all be the same
    println("wid.home graph_s, home_id: ", wid_df.sp_hh_id[1:5])
    println("col(people_in_homes_df): ", names(people_in_homes_df))
    println("people_in_homes_df.home_id: ", people_in_homes_df.sp_hh_id)
    println("people_in_homes_df.work_id: ", people_in_homes_df.work_id)
    println("")
    # home_id should also be in people_in_homes_df
    set_prop!(graph_w, :ncoords_w, [wlong, wlat,   fill(wplane, length(wlong))])
    set_prop!(graph_w, :ncoords_h, [hwlong, hwlat, fill(hplane, length(hwlong))])

    return [graph_s, graph_w]
end

function secondGeneration!(scene, graph_s, graph_w; marker_size=0.01)
    xs, ys, zs = get_prop(graph_s, :ncoords_s);
    xh, yh, zh = get_prop(graph_s, :ncoords_h);
    #home_sxy = [Set(xh), Set(yh)]
    #println("maximum coord_h: ", maximum.([xh, yh, zh]))
    #println("maximum coord_s: ", maximum.([xs, ys, zs]))
    xsegs = zeros(2*nv(graph_s))
    ysegs = zeros(2*nv(graph_s))
    zsegs = zeros(2*nv(graph_s))

    for i in 1:nv(graph_s)
        xsegs[2*i-1] = xh[i]
        xsegs[2*i]   = xs[i]
        ysegs[2*i-1] = yh[i]
        ysegs[2*i]   = ys[i]
        zsegs[2*i-1] = zh[i]
        zsegs[2*i]   = zs[i]
    end

    w_col = :red
    linesegments!(scene, xsegs, ysegs, zsegs, color=w_col, edgewidth=1.)
    AP.scatter!(scene, xs ,ys, zs, transparency=true, markersize=4*marker_size, color=w_col)
    AP.scatter!(scene, xh ,yh, zh, transparency=true, markersize=marker_size, color=:darkgray)
    #meshscatter!(scene, xs ,ys, zs, transparency=true, markersize=4*marker_size, color=w_col)
    #meshscatter!(scene, xh ,yh, zh, transparency=true, markersize=marker_size, color=:darkgray)

    xw, yw, zw = get_prop(graph_w, :ncoords_w);
    xh, yh, zh = get_prop(graph_w, :ncoords_h);
    #home_wxy = [Set(xh), Set(yh)]

    #println("maximum coord_h: ", maximum.([xh, yh, zh]))
    #println("maximum coord_w: ", maximum.([xw, yw, zw]))
    xsegs = zeros(2*nv(graph_w))
    ysegs = zeros(2*nv(graph_w))
    zsegs = zeros(2*nv(graph_w))

    for i in 1:nv(graph_w)
        xsegs[2*i-1] = xh[i]
        xsegs[2*i]   = xw[i]
        ysegs[2*i-1] = yh[i]
        ysegs[2*i]   = yw[i]
        zsegs[2*i-1] = zh[i]
        zsegs[2*i]   = zw[i]
    end

    sch_col = :blue
    linesegments!(scene, xsegs, ysegs, zsegs, color=sch_col)
    AP.scatter!(scene, xw ,yw, zw, transparency=true, markersize=4*marker_size, color=sch_col, alpha=.0, fillalpha=.0, markeralpha=0.0)
    AP.scatter!(scene, xh ,yh, zh, transparency=true, markersize=marker_size, color=:darkgray, alpha=.0, fillalpha=.0, markeralpha=0.0)
    #meshscatter!(scene, xw ,yw, zw, transparency=true, markersize=4*marker_size, color=sch_col, alpha=.0, fillalpha=.0, markeralpha=0.0)
    #meshscatter!(scene, xh ,yh, zh, transparency=true, markersize=marker_size, color=:darkgray, alpha=.0, fillalpha=.0, markeralpha=0.0)
    return
end

function setupCamera!(scene)
    cam = cam3d!(scene)
    #cam.projectiontype[] = AbstractPlotting.Orthographic
    #cam.
    eyepos = Makie.Vec3f0(5, 1.5, 0.5);
    lookat = Makie.Vec3f0(0., 0., 0.);
end

# Plot Leon county homes (white), businesses (green), schools (blue) in a single plane
function plotCounty!(scene, all_df; county_plane=0.0, marker_size=0.0001)
    house_xy = Matrix(all_df[[:hlong, :hlat]])
    house_xy[:,1] .+= 84.
    house_xy[:,2] .-= 30.
    house_z = repeat([county_plane], nrow(all_df))  # waste of memory
    print("house_z: ", house_z[1:10])
    AP.scatter!(scene, house_xy[:,1], house_xy[:,2], house_z,
        strokecolor=:gray, markersize=marker_size,color=:gray)
end

# ----------------------------------------------
# Remove while debugging
# the first node is the workplace, the others, the home
const hh = 3.
hplane = 0.6
wplane = 0.66
countyplane = 0.598


graph, wid_df = randomBusinessWorkToHomes(all_df, df_w, 50, 70)
xx, yy = get_prop(graph, :node_coords)
mgraph = collectGraphEdges(graph, hplane=hplane, wplane=wplane);
graph_s, graph_w = getHomes(wid_df, hplane=hplane, wplane=wplane)

# Could run in a separate thread
# Empty scene takes a long time to create. Weird.

makePlot(mgraph, graph_s, graph_w)
nothing

function makePlot(mgraph, graph_s, graph_w)
    marker_size = 0.0002
    scene = Scene();
    s1, radius = textslider(0.0f0:.4f0:4.0f0, "Radius", start = 2.6f0)
    rad = to_value(radius)
    x,y,z = get_prop(mgraph, :line_segments);
    AP.linesegments!(scene, x,y,z, color=:darkblue);
    #linesegments!(scene, x,y,z, color=:darkblue);
    AP.scatter!(scene, x[1:1], y[1:1], z[1:1], markersize=30*marker_size, color=:darkblue, transparency=false)
    x,y,z = get_prop(mgraph, :ncoords);

    # returns home coordinates connected to workplaces and homes as Sets
    secondGeneration!(scene, graph_s, graph_w,
         marker_size=20*marker_size)

    plotCounty!(scene, all_df, county_plane=countyplane, marker_size=marker_size)

    Point3f0 = GeometryTypes.Point3f0
    #sphere = HyperSphere(Point3f0(0), to_value(radius)*1f0)
    sphere = HyperSphere(Point3f0(0), 1f0)
    positions = GeometryTypes.decompose(Point3f0, sphere)
    #view(positions, rand(1:length(positions), 100))
    println("rad= ", rad)
    AP.meshscatter!(scene, positions, markersize=rad*.2, color=:blue, transparency=false)
    #scatter!(scene, positions, transparency=true, strokewidth=.02, strokecolor=:blue, color=:blue)

    parent_scene = Scene(resolution=(900, 900))

    #hbox(vbox(s1), vbox(scene), parent=parent_scene)
    vbox(hbox(s1, scene), parent=parent_scene)
    setupCamera!(scene);
    display(parent_scene)
end
nothing

# ----------------------------------------------------------------------

hsx, hsy = home_sxy
hwx, hwy = home_wxy
println(typeof(home_sxy[1]))
println(home_sxy[1])
println(home_sxy[2])
wid_hx = Set(wid_df.hlong.+84.)
wid_hy = Set(wid_df.hlat.-30.)
nothing



# ----------------------------------------------------------------------
# ======================================================================
# ======================================================================
# LAST FUNCTION

#function getHomes(wid_df; hplane=0., wplane=1.)
    # Starting from the homes, Identify all the workplaces.
    # identify the home coordinates of all the workers
println("hlat people: ", people_in_homes_df.hlat .-30.)

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
#end

function Gordon(a::Real)
    println(a)
end
