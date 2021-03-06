module EnsembleKalmanProcesses

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

#auxilliaries
include("ParameterDistribution.jl")
include("DataStorage.jl")
include("Observations.jl")

#updates:
include("EnsembleKalmanProcess.jl")
end # module
