module EnsembleKalmanProcesses

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# auxiliary modules
include("ParameterDistributions.jl")
include("DataContainers.jl")
include("Observations.jl")
include("Localizers.jl")
include("UQParameters.jl")

# algorithmic updates
include("EnsembleKalmanProcess.jl")

end # module
