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
using JLD2

###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn
@everywhere param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
n_param = length(param_names)

# Prior information: Define transform to unconstrained gaussian space
constraints = [ [bounded(0.01, 0.3)],
                [bounded(0.01, 0.9)],
                [bounded(0.25, 4.0)],
                [bounded(0.01, 0.5)],
                [bounded(0.01, 0.5)],
                [bounded(0.0, 0.5)],
                [bounded(0.0, 0.5)],
                [bounded(5.0, 15.0)],
                [bounded(0.1, 0.8)]]
# All vars are standard Gaussians in unconstrained space
prior_dist = [Parameterized(Normal(0.0, 0.5))
                for x in range(1, n_param, length=n_param) ]
priors = ParameterDistribution(prior_dist, constraints, param_names)
@everywhere priors = $priors

###
###  Retrieve true LES samples from PyCLES data
###

# Define observation window per flow condition
@everywhere ti = [7200.0, 25200.0, 7200.0, 14400.0]
@everywhere tf = [14400.0, 32400.0, 14400.0, 21600.0]
# Define variables per flow condition
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex
@everywhere y_names=$y_names

# Define preconditioning of inverse problem
@everywhere normalized = true
@everywhere perform_PCA = true
@everywhere pool_norm = false
@everywhere eigval_norm = false

@everywhere sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]
@everywhere sim_suffix = [".may20", ".iles128wCov", ".dry11", ".may18"]
# Init arrays
yt = zeros(0)
yt_var_list = []
yt_big = zeros(0)
yt_var_list_big = []
P_pca_list = []
pool_var_list = []
for (i, sim_name) in enumerate(sim_names)
    if occursin("Nieuwstadt", sim_name)
        les_dir = string("/groups/esm/ilopezgo/Output.", "Soares", sim_suffix[i])
    else
        les_dir = string("/groups/esm/ilopezgo/Output.", sim_name, sim_suffix[i])
    end
    # Get SCM vertical levels for interpolation
    z_scm = get_profile(string("Output.", sim_name, ".00000"), ["z_half"])
    # Get (interpolated and pool-normalized) observations, get pool variance vector
    yt_, yt_var_, pool_var = obs_LES(y_names[i], les_dir, ti[i], tf[i], z_scm = z_scm, normalize=normalized)
    push!(pool_var_list, pool_var)
    if perform_PCA
        yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_, 1.0e-2,
            eigval_norm = eigval_norm, pool_norm = pool_norm)
        append!(yt, yt_pca)
        push!(yt_var_list, yt_var_pca)
        push!(P_pca_list, P_pca)
    else
        append!(yt, yt_)
        push!(yt_var_list, yt_var_)
        global P_pca_list = nothing
    end
    # Save full dimensionality (normalized) output for error computation
    append!(yt_big, yt_)
    push!(yt_var_list_big, yt_var_)
end

@everywhere yt = $yt
@everywhere yt_var_list = $yt_var_list
@everywhere yt_big = $yt_big
@everywhere yt_var_list_big = $yt_var_list_big
@everywhere P_pca_list = $P_pca_list
@everywhere N_yt = length(yt) # Length of data array
@everywhere pool_var_list = $pool_var_list

# Construct global observational covariance matrix
yt_var = zeros(N_yt, N_yt)
vars_num = 1
for flow_cov in yt_var_list
    vars = length(flow_cov[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = flow_cov
    global vars_num = vars_num+vars
    println("DETERMINANT OF PCA OBS NOISE COV MATRIX FOR 1 FLOW, ", det(flow_cov))
end
@everywhere yt_var = $yt_var

n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
# Regularization nugget
@everywhere noise_level = 1.0
@everywhere Γy = noise_level * Matrix(1.0I, N_yt, N_yt) + yt_var
println("DETERMINANT OF FULL OBS NOISE COV MATRIX, ", det(Γy))

###
###  Calibrate: Ensemble Kalman Inversion
###

@everywhere N_ens = 50 # number of ensemble members
@everywhere N_iter = 10 # number of EKP iterations.

initial_params = construct_initial_ensemble(priors, N_ens)
# Discard unstable parameter combinations
precondition_ensemble!(initial_params, priors, param_names, y_names, ti, tf=tf)
@everywhere initial_params = $initial_params

@everywhere ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, Inversion())
@everywhere scm_dir = "/home/ilopezgo/SCAMPy/"
@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names,
   y_names, scm_dir, ti, tf, P_pca_list = P_pca_list, norm_var_list = pool_var_list)

# Create output dir
prefix = perform_PCA ? "results_pycles_PCA_" : "results_pycles_" # = true
prefix = pool_norm ? string(prefix, "pooled_") : prefix
prefix = eigval_norm ? string(prefix, "eignorm_") : prefix
outdir_path = string(prefix, "p", n_param,"_n", noise_level,"_e", N_ens, "_i", N_iter, "_d", N_yt)
println("Name of outdir path for this EKP, ", outdir_path)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# EKP iterations
g_ens = zeros(N_ens, N_yt)
@everywhere Δt = 1.0
for i in 1:N_iter
    # Note that the parameters are exp-transformed when used as input
    # to SCAMPy
    @everywhere params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, 
        get_u_final(ekobj)) )
    @everywhere params = [row[:] for row in eachrow(params_cons_i')]
    g_ens_arr, g_ens_arr_pca = pmap(g_, params)
    println(string("\n\nEKp evaluation ",i," finished. Updating ensemble ...\n"))
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr_pca[j]
    end
    update_ensemble!(ekobj, Array(g_ens') )
    println("\nEnsemble updated. Saving results to file...\n")
    # Save EKP information to file
    save(string(outdir_path,"/ekp.jld2"),
        "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
        "ekp_g", get_g(ekobj),
        "truth_mean", ekobj.obs_mean,
        "truth_cov", ekobj.obs_noise_cov,
        "ekp_err", ekobj.err,
        "norm_err", compute_errors(g_ens_arr, y))
end

# EKP results: Has the ensemble collapsed toward the truth?
println("\nEKP ensemble mean at last stage (original space):")
println(mean(transform_unconstrained_to_constrained(priors, get_u(ekobj)), dims=1))

println("\nEnsemble covariance det. 1st iteration, transformed space.")
println(det( cov((get_u(ekobj, 1)), dims=1) ))
println("\nEnsemble covariance det. last iteration, transformed space.")
println(det( cov(get_u_final(ekobj), dims=2) ))
