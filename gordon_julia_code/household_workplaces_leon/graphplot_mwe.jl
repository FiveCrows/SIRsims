using LightGraphs
using GraphRecipes
import Plots

Plots.PlotlyBackend()
#Plots.plotly()

function plotGraph()
    graph = SimpleGraph(200, 1)
    nb_nodes = nv(graph)
    xx = rand(nb_nodes)
    yy = rand(nb_nodes)
    zz = rand(nb_nodes)
    @show graph
    # ERROR. Also, when curves=false, there is an error
    p = graphplot(graph, method=nothing, curvature=0, dim=3, x=xx, y=yy, z=zz)#, curves=false)
    #p = graphplot(graph, curvature=0, dim=2, curves=false)
end

p = plotGraph()
# drawing takes a long time even with no edges.
# edges display correctly in 2D, yet the graph takes a long time to display
# it is as if there is lots of compilation going on every time the graph is plotted.
# WHY IS THAT?


Plots.GR()

function plotGraph(graph, x, y, z)
    edges_ = []
    nb_nodes = nv(graph)
    nb_edges = ne(graph)

    p = Plots.plot([],[])

    xx = zeros(2)
    yy = zeros(2)
    zz = zeros(2)
    for e in edges(graph)
        i = src(e)
        j = dst(e)
        xx[:] = [x[i], x[j]]
        yy[:] = [y[i], y[j]]
        zz[:] = [z[i], z[j]]
        if i == 1
            p = Plots.plot(xx,yy,zz, color=:red, leg=false)
        else
            @time Plots.plot!(p,xx,yy,zz, color=:red, leg=false)
        end
    end
    return p
end

const n = 1000
const graph = SimpleGraph(n, 3*n)

const x = rand(n)
const y = rand(n)
const z = rand(n)
@time const p = plotGraph(graph, x, y, z);
@time display(p)

using Makie
AbstractPlotting.inline!(false)
scene = Scene()

n = 1000
x = rand(n)
y = rand(n)
z = rand(n)
colors = rand(n)
@time scene = scatter3d(x, y, z, color = colors, markersize=.1)
@time display(scene)
