module EnsembleKalmanProcesses

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# auxiliary modules
include("ParameterDistributions.jl")
include("DataContainers.jl")
include("Observations.jl")
include("Localizers.jl")
include("TOMLInterface.jl")

# algorithmic updates
include("EnsembleKalmanProcess.jl")

# Plot recipes
include("PlotRecipes.jl")
end # module
