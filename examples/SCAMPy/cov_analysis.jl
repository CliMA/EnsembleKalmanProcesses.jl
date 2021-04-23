# Import modules
using Pkg
Pkg.activate(".")
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "helper_funcs.jl"))
using JLD
using NPZ

###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn

param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
n_param = length(param_names)

# Assume lognormal priors for all parameters
# Note: For the EDMF model to run, all parameters need to be nonnegative. 
# The EKp update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e.,
# (the parameters can then simply be obtained by exponentiating the final results). 

# Prior: Log-normal in original space defined by mean and std
logmeans = zeros(n_param)
log_stds = zeros(n_param)
logmeans[1], log_stds[1] = logmean_and_logstd(0.2, 0.2)
logmeans[2], log_stds[2] = logmean_and_logstd(0.4, 0.2)
logmeans[3], log_stds[3] = logmean_and_logstd(2.0, 1.0)
logmeans[4], log_stds[4] = logmean_and_logstd(0.2, 0.2)
logmeans[5], log_stds[5] = logmean_and_logstd(0.2, 0.2)
logmeans[6], log_stds[6] = logmean_and_logstd(0.2, 0.2)
logmeans[7], log_stds[7] = logmean_and_logstd(0.2, 0.2)
logmeans[8], log_stds[8] = logmean_and_logstd(8.0, 1.0)
logmeans[9], log_stds[9] = logmean_and_logstd(0.2, 0.2)
prior_dist = [Parameterized(Normal(logmeans[1], log_stds[1])),
                        Parameterized(Normal(logmeans[2], log_stds[2])),
                        Parameterized(Normal(logmeans[3], log_stds[3])),
                        Parameterized(Normal(logmeans[4], log_stds[4])),
                        Parameterized(Normal(logmeans[5], log_stds[5])),
                        Parameterized(Normal(logmeans[6], log_stds[6])),
                        Parameterized(Normal(logmeans[7], log_stds[7])),
                        Parameterized(Normal(logmeans[8], log_stds[8])),
                        Parameterized(Normal(logmeans[9], log_stds[9]))]

###
###  Retrieve true LES samples from PyCLES data
###

# This is the true value of the observables (e.g. LES ensemble mean for EDMF)
ti = [10800.0, 28800.0, 10800.0, 18000.0]
tf = [14400.0, 32400.0, 14400.0, 21600.0]
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex

# Get observations
yt = zeros(0)
sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]
yt_var_list = []

les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[1],".may20")
sim_dir = string("Output.", sim_names[1],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[1], les_dir, ti[1], tf[1], z_scm = z_scm)
append!(yt, yt_)
push!(yt_var_list, yt_var_)
npzwrite("dycoms_z.npy", z_scm)

les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[2],".iles128wCov")
sim_dir = string("Output.", sim_names[2],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[2], les_dir, ti[2], tf[2], z_scm = z_scm)
append!(yt, yt_)
push!(yt_var_list, yt_var_)
npzwrite("gabls_z.npy", z_scm)

les_dir = string("/groups/esm/ilopezgo/Output.Soares.dry11")
sim_dir = string("Output.", sim_names[3],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[3], les_dir, ti[3], tf[3], z_scm = z_scm)
append!(yt, yt_)
push!(yt_var_list, yt_var_)
npzwrite("nieuwstadt_z.npy", z_scm)

les_dir = string("/groups/esm/ilopezgo/Output.Bomex.may18")
sim_dir = string("Output.", sim_names[4],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[4], les_dir, ti[4], tf[4], z_scm = z_scm)
append!(yt, yt_)
push!(yt_var_list, yt_var_)
npzwrite("bomex_z.npy", z_scm)

yt_var = zeros(length(yt), length(yt))
vars_num = 1
for sim_covmat in yt_var_list
    vars = length(sim_covmat[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = sim_covmat
    global vars_num = vars_num+vars
    #println(det(sim_covmat))
end
println( det(yt_var))
n_observables = length(yt)

# This is how many samples of the true data we have
n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
# Noise level of the samples, which scales the time variance of each output.
noise_level = 1.0
Γy = noise_level^2 * (yt_var)
μ_noise = zeros(length(yt))

save("obs_cov.jld", "cov_mat", yt_var)
npzwrite("obs_cov.npy", yt_var)

