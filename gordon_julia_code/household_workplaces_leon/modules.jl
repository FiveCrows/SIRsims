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
include("./household_workplace_functions.jl")
# Not sure about the meaning of dot. Does not work with out. Means current directory?
using .FastSIRWithWeightedNodes
FWG = FastSIRWithWeightedNodes

# import makes sure that the namespace prefix is used for safety since
# method names are the sme s in FastSIR_impl.jl
include("FastSIR_multiplex_impl.jl")
import .FastSIRMultiplex
FPLEX = FastSIRMultiplex

include("./FastSIR_common.jl")
using .FastSIRCommon
F  = FastSIRCommon
