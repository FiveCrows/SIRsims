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

@time times, S,I,R = FPLEX.fastSIR(Glist, p, collect(1:2000); t_max=100.)
F.myPlot(times, S, I, R)

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
