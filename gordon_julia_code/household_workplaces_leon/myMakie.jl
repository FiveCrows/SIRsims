using Makie
using AbstractPlotting
using BenchmarkTools
using LightGraphs, MetaGraphs

n = 30000
x = 1:n
y = rand(n)
@time scene = lines(x,y)
#@time display(scene)


x = [1,2,NaN,6,7,NaN,10,12];
y = [1,2,3,4,5,6,7,8];
z = rand(length(x));
x = repeat(x,1000);
y = repeat(y,1000);
z = repeat(z,1000);
@time scene = lines(x,y,z);

function testGraph!(scene; zplane::Float32=0.0f0)
    n = 1000000
    n = 10000
    n = 100000
    n = 10000
    n = 400
    graph = SimpleGraph(n, 3*n)
    nes = ne(graph)
    xn = rand(n)
    yn = rand(n)
    zn = fill(zplane, n)
    xe = zeros(Float32, 2*nes)
    ye = zeros(Float32, 2*nes)
    ze = fill(zplane, 2*nes)
    for (i,e) in enumerate(edges(graph))
        xe[2*i-1] = xn[src(e)]
        xe[2*i]   = xn[dst(e)]
        ye[2*i-1] = yn[src(e)]
        ye[2*i]   = yn[dst(e)]
        ze[2*i-1] = zn[src(e)]
        ze[2*i]   = zn[dst(e)]
    end
    mgraph = MetaGraph(graph)
    set_prop!(mgraph, :ncoords, [xn, yn, zn])
    set_prop!(mgraph, :ecoords, [xe, ye, ze])
    return mgraph
end

function drawGraph!(scene, mgraph::MetaGraph; edge_color=:black)
    xn, yn, zn = get_prop(mgraph, :ncoords)
    xe, ye, ze = get_prop(mgraph, :ecoords)
    scatter!(scene, xn, yn, zn, color=edge_color, markersize=.02)
    linesegments!(scene, xe, ye, ze, color=edge_color)
end

# Randomly connect the two graphs
function connectGraphs(g1::MetaGraph, g2::MetaGraph, nb_edges::Int)
    n1 = nv(g1)
    n2 = nv(g2)
    x1,y1,z1 = get_prop(g1, :ncoords)  # 3-vec
    x2,y2,z2 = get_prop(g2, :ncoords)
    ix = rand(1:nb_edges, nb_edges)
    xe = zeros(Float32, 2*nb_edges)
    ye = zeros(Float32, 2*nb_edges)
    ze = zeros(Float32, 2*nb_edges)
    for i in 1:nb_edges
        xe[2i-1] = x1[ix[i]]
        xe[2i]   = x2[ix[i]]
        ye[2i-1] = y1[ix[i]]
        ye[2i]   = y2[ix[i]]
        ze[2i-1] = z1[ix[i]]
        ze[2i]   = z2[ix[i]]
    end
    return xe, ye, ze
end


# I really want an empty scene
# If I draw the markers in white, the lines in testGraph are drawn in white. Strange.
graph1 = testGraph!(scene, zplane=0.250f0);
graph2 = testGraph!(scene, zplane=0.75f0) ;
xe, ye, ze = connectGraphs(graph1, graph2, 100)

# Draw Graphs
scene = scatter([0.], [0.], [0.], markersize=.01, color=:red)
drawGraph!(scene, graph1; edge_color=:blue)
linesegments!(scene, xe, ye, ze, color=:green)
drawGraph!(scene, graph2, edge_color=:red)


cam = cam3d!(scene)
cam.projectiontype[] = AbstractPlotting.Orthographic
cam.upvector = Vec3f0(0,0,1)
eyepos = Vec3f0(5, 1.5, 0.5);
lookat = Vec3f0(0);
update_cam!(scene, eyepos, lookat);
