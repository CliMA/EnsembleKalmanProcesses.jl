using Distributions
using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
include(joinpath(@__DIR__, "helper_funcs.jl"))




run(`mkdir -p output`)
run(`mkdir -p output/slurm`)
for i = 1:N_ens
    run(`mkdir -p $(input_prefix)$(i)`)
    run(`mkdir -p $(output_prefix)$(i)`)
end

