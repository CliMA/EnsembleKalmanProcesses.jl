using Distributions
using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
include(joinpath(@__DIR__, "helper_funcs.jl"))

# Set parameter priors
param_names = ["C_smag", "C_drag"]
n_param = length(param_names)
prior_dist = [Parameterized(Normal(0.5, 0.05)), Parameterized(Normal(0.001, 0.0001))]
constraints = [[no_constraint()], [no_constraint()]]
priors = ParameterDistribution(prior_dist, constraints, param_names)

# Construct initial ensemble
N_ens = 10
initial_params = construct_initial_ensemble(priors, N_ens)
# Generate CLIMAParameters files
params_arr = [row[:] for row in eachrow(initial_params')]
versions = map(param -> generate_cm_params(param, param_names), params_arr)

# Store version identifiers for this ensemble in a common file
open("versions_1.txt", "w") do io
    for version in versions
        write(io, "clima_param_defs_$(version).jl\n")
    end
end
