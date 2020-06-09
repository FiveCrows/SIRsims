module  FastSIR
using LightGraphs
import Random: seed!
using Distributions
using DataStructures

# Exports come before indlude statements
export watch

include("FastSIR_impl.jl");


end
