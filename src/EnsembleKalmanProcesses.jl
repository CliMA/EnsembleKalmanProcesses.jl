module EnsembleKalmanProcesses

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

#auxilliaries
include("ParameterDistribution.jl")
include("DataStorage.jl")

#updates:
include("EnsembleKalmanProcess.jl")

end # module
