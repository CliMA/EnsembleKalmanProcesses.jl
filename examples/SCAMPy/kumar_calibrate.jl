# Import modules
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Distributions  # probability distributions and associated functions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "helper_funcs.jl"))
using JLD
@everywhere using NPZ
###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn

@everywhere param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
@everywhere n_param = length(param_names)

# Assume lognormal priors for all parameters
# Note: For the EDMF model to run, all parameters need to be nonnegative. 
# The EKI update can result in violations of 
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
@everywhere prior_dist = $prior_dist

###
###  Retrieve true LES samples from PyCLES data
###
# This is the true value of the observables (e.g. LES horizontal mean)
@everywhere data_dir = "/groups/esm/ilopezgo/padeops_data/"
@everywhere padeops_t = npzread( string(data_dir,"time_padeops.npy") )*3600.0
@everywhere padeops_z = npzread( string(data_dir,"zCell_padeops.npy") )*1000.0
@everywhere padeops_theta = npzread( string(data_dir,"potT_padeops.npy") )
@everywhere padeops_uh = npzread( string(data_dir,"wind_speed_padeops.npy") )

# Times on which to interpolate
@everywhere t_fig3 = ([4, 6, 8, 10, 12, 15.5, 17, 18.5, 20, 21.5])*3600.0
@everywhere t_fig5b = ([6, 7, 8, 9, 10, 11, 12, 13])*3600.0
# Get SCM vertical grid
@everywhere sim_names = ["Kumar_dc_init_m1"]
sim_dir = string("Output.", sim_names[1],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
# Initialize objectives
@everywhere yt = zeros(0)
yt_var_list = []

yt_, yt_var_ = padeops_m_σ2(padeops_theta, padeops_z, padeops_t, z_scm, t_fig3)
append!(yt, yt_)
push!(yt_var_list, yt_var_)
yt_, yt_var_ = padeops_m_σ2(padeops_uh, padeops_z, padeops_t, z_scm, t_fig3)
append!(yt, yt_)
push!(yt_var_list, yt_var_)

@everywhere yt = $yt
yt_var = zeros(length(yt), length(yt))
vars_num = 1
for sim_covmat in yt_var_list
    vars = length(sim_covmat[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = sim_covmat
    global vars_num = vars_num+vars
end
@everywhere yt_var = $yt_var
@everywhere n_observables = length(yt)
padeops_names = Array{String, 1}[]
push!(padeops_names, ["theta_fig3", "uh_fig5b"])
@everywhere padeops_names=$padeops_names

# This is how many samples of the true data we have
n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
# Noise level of the samples, which scales the time variance of each output.
noise_level = 1.0
Γy = noise_level^2 * (yt_var)
μ_noise = zeros(length(yt))
# We construct the observations object with the samples and the cov.
truth = Obs(Array(samples'), Γy, padeops_names[1])
@everywhere truth = $truth


###
###  Calibrate: Ensemble Kalman Inversion
###

@everywhere N_ens = 10 # number of ensemble members
@everywhere N_iter = 3 # number of EKI iterations.
@everywhere N_yt = length(yt) # Length of data array

@everywhere constraints = [[no_constraint()], [no_constraint()],
                [no_constraint()], [no_constraint()],
                [no_constraint()], [no_constraint()],
                [no_constraint()], [no_constraint()],[no_constraint()]]

@everywhere priors = ParameterDistribution(prior_dist, constraints, param_names)
@everywhere initial_params = construct_initial_ensemble(priors, N_ens)
y_names = ["thetal_mean", "horizontal_vel"]
precondition_ensemble!(Array(initial_params'), priors, param_names, y_names, t_fig3)
@everywhere initial_params = $initial_params

@everywhere ekobj = EnsembleKalmanProcess(initial_params, yt, yt_var, Inversion()) 

g_ens = zeros(N_ens, n_observables)

@everywhere scm_dir = "/home/ilopezgo/SCAMPy/"
@everywhere params_i = deepcopy(exp_transform(get_u_final(ekobj)))

@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names,
   y_names, scm_dir, t_fig3)

outdir_path = string("results_p", n_param,"_n", noise_level,"_e", N_ens, "_i", N_iter, "_d", N_yt)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# EKI iterations
@everywhere Δt = 1.0
for i in 1:N_iter
    # Note that the parameters are exp-transformed when used as input
    # to SCAMPy
    @everywhere params_i = deepcopy(exp_transform(get_u_final(ekobj)))
    # @everywhere params_i = [params_i[i, :] for i in 1:size(params_i, 1)]
    @everywhere params = [row[:] for row in eachrow(params_i')]
    g_ens_arr = pmap(g_, params)
    println(string("\n\nEKI evaluation ",i," finished. Updating ensemble ...\n"))
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr[j]
    end
    update_ensemble!(ekobj, Array(g_ens') )
    println("\nEnsemble updated.\n")
    # Save EKI information to file
    save( string(outdir_path,"/eki.jld"), "eki_u", ekobj.u, "eki_g", ekobj.g,
        "truth_mean", ekobj.obs_mean, "truth_cov", ekobj.obs_noise_cov, "eki_err", ekobj.err)
end

# Save EKI information to file
save("eki.jld", "eki_u", ekobj.u, "eki_g", ekobj.g,
        "truth_mean", ekobj.obs_mean, "truth_cov", ekobj.obs_noise_cov, "eki_err", ekobj.err)

# EKI results: Has the ensemble collapsed toward the truth? Store and analyze.
println("\nEKI ensemble mean at last stage (original space):")
println(mean(deepcopy(exp_transform(get_u_final(ekobj) )), dims=1))

println("\nEnsemble covariance det. 1st iteration, transformed space.")
println(det(cov(deepcopy((get_u(ekobj, 1) )), dims=1)))

println("\nEnsemble covariance det. last iteration, transformed space.")
println(det(cov(deepcopy((get_u_final(ekobj) )), dims=1)))
