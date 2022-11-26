using Distributions
using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
include(joinpath(@__DIR__, "helper_funcs.jl"))




run(`mkdir -p output`)
run(`cd output`)
for i = 1:N_ens
    run(`mkdir -p output_fms_$(i)`)
end

