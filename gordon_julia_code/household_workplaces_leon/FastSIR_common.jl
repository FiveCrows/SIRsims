module FastSIRCommon
using LightGraphs
using Plots

function makeGraph(nb_nodes, edges_per_vertex)
    random_regular_digraph(nb_nodes, edges_per_vertex)
end

function myPlot(times, S, I, R)
   plot(times,  S, label=":S") # symbol legends
   plot!(times, I, label=":I") # used to work
   plot!(times, R, label=":R")
end

# https://stackoverflow.com/questions/58733735/how-to-plot-heatmap-in-julia
function makeHeatMap(x,y,f)
    heatmap(1:size(f),
        1:size(x), data,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        title="My title")
end

function makeHeatMap(data)
    heatmap(1:size(data,1),
        1:size(data,2), data,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x values", ylabel="y values",
        title="My title")
end

end
