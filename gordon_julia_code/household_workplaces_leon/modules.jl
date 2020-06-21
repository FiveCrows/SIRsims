using BenchmarkTools
using LightGraphs
using SimpleWeightedGraphs
using MetaGraphs
using Distributions
using CSV
categ = Distributions.Categorical
using Random
import StatsBase: sample, sample!

using DataFrames
DF = DataFrames

include("FastSIR_impl.jl")
# Not sure about the meaning of dot. Does not work with out. Means current directory?
using .FastSIRWithWeightedNodes
FWG = FastSIRWithWeightedNodes

include("./FastSIR_common.jl")
using .FastSIRCommon
F  = FastSIRCommon

include("./household_workplace_functions.jl")
