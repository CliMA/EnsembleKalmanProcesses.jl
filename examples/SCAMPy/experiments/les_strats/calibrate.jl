# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))
using JLD2
using NPZ

#########
#########  Define the parameters and their priors
#########

# Define the parameters that we want to learn
param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
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


#########
#########  Retrieve true LES samples from PyCLES data and transform
#########

# Define observation window per flow configuration
ti = [7200.0, 25200.0, 7200.0, 14400.0]
tf = [14400.0, 32400.0, 14400.0, 21600.0]
# Define variables per flow configuration
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex

# Define preconditioning and regularization of inverse problem
normalized = true # Variable normalization
perform_PCA = true # PCA on config covariance
cutoff_reg = true # Regularize above PCA cutoff
beta = 10.0 # Regularization hyperparameter
variance_loss = 1.0e-1 # PCA variance loss
noisy_obs = true # Choice of covariance in evaluation of y_{j+1} in EKI. True -> Γy, False -> 0

sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]
sim_suffix = [".may20", ".iles128wCov", ".dry11", ".may18"]
# Init arrays
yt = zeros(0)
yt_var_list = []
yt_big = zeros(0)
yt_var_list_big = []
P_pca_list = []
pool_var_list = []
for (i, sim_name) in enumerate(sim_names)
    les_dir = occursin("Nieuwstadt", sim_name) ? 
        string("/groups/esm/ilopezgo/Output.", "Soares", sim_suffix[i]) : string("/groups/esm/ilopezgo/Output.", sim_name, sim_suffix[i])
    # Get SCM vertical levels for interpolation
    z_scm = get_profile(string("Output.", sim_name, ".00000"), ["z_half"])
    # Get (interpolated and pool-normalized) observations, get pool variance vector
    yt_, yt_var_, pool_var = obs_LES(y_names[i], les_dir, ti[i], tf[i], z_scm = z_scm, normalize=normalized)
    push!(pool_var_list, pool_var)
    if perform_PCA
        yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_, variance_loss)
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

d = length(yt) # Length of data array
# Construct global observational covariance matrix for error diagnostic, no TSVD
yt_var_big = zeros(length(yt_big), length(yt_big))
vars_num = 1
for (k,config_cov) in enumerate(yt_var_list_big)
    vars = length(config_cov[1,:])
    yt_var_big[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = config_cov
    global vars_num = vars_num+vars
    println("DETERMINANT OF Γy FOR ", sim_names[k], " ", det(config_cov))
end

# Construct global observational covariance matrix for EKP, TSVD
yt_var = zeros(d, d)
vars_num = 1
for (k,config_cov) in enumerate(yt_var_list)
    vars = length(config_cov[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = cutoff_reg ? 
         (config_cov + (beta*minimum(diag(config_cov))) .* Matrix(1.0I, size(config_cov)) ) .* vars : config_cov .* vars
    global vars_num = vars_num+vars
    println("DETERMINANT OF PCA Γy FOR ", sim_names[k], " ", det(config_cov))
end
println("NUMBER OF OUTPUTS CONSIDERED ", d, " CAPTURING FRACTION OF VARIANCE: ", 1.0-variance_loss)

samples = zeros(1, length(yt))
samples[1,:] = yt
Γy = yt_var
println("DETERMINANT OF FULL Γy, ", det(Γy))


#########
#########  Calibrate: Ensemble Kalman inversion
#########

algo = Unscented(vcat(get_mean(priors)...), get_cov(priors), length(yt), 1.0, 0 ) # Sampler(vcat(get_mean(priors)...), get_cov(priors)) # Inversion()
N_ens = typeof(algo) == Unscented{Float64,Int64} ? 2*n_param + 1 : 50 # number of ensemble members
N_iter = 10 # number of EKP iterations.
Δt = 1.0/length(sim_names)
println("NUMBER OF ENSEMBLE MEMBERS: ", N_ens)
println("NUMBER OF ITERATIONS: ", N_iter)
deterministic_forward_map = noisy_obs ? true : false

initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
# Discard unstable parameter combinations, parallel
#precondition_ensemble!(initial_params, priors, param_names, y_names, ti, tf=tf)

# UKI does not require sampling from the prior
ekobj = typeof(algo) == Unscented{Float64,Int64} ?  EnsembleKalmanProcess(yt, Γy, algo ) : EnsembleKalmanProcess(initial_params, yt, Γy, algo )
scm_dir = "/home/ilopezgo/SCAMPy/"
@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, $param_names,
   $y_names, $scm_dir, $ti, $tf, P_pca_list = $P_pca_list, norm_var_list = $pool_var_list) 

# Create output dir
prefix = perform_PCA ? "results_pycles_PCA_" : "results_pycles_" # = true
prefix = cutoff_reg ? string(prefix, "creg", beta, "_") : prefix
prefix = typeof(algo) == Sampler{Float64} ? string(prefix, "eks_") : prefix
prefix = typeof(algo) == Unscented{Float64,Int64} ? string(prefix, "uki_") : prefix
prefix = noisy_obs ? prefix : string(prefix, "nfo_")
prefix = Δt ≈ 1 ? prefix : string(prefix, "dt", Δt, "_")
outdir_path = string(prefix, "p", n_param,"_n", noise_level,"_e", N_ens, "_i", N_iter, "_d", d)
println("Name of outdir path for this EKP, ", outdir_path)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# EKP iterations
g_ens = zeros(N_ens, d)
norm_err_list = []
g_big_list = []
for i in 1:N_iter
    # Note that the parameters are transformed when used as input to SCAMPy
    params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, 
        get_u_final(ekobj)) )
    params = [row[:] for row in eachrow(params_cons_i')]
    @everywhere params = $params
    array_of_tuples = pmap(g_, params) # Outer dim is params iterator
    (g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),2) # Outer dim is G̃, G 
    println("LENGTH OF G_ENS_ARR", length(g_ens_arr))
    println("LENGTH OF G_ENS_ARR_PCA", length(g_ens_arr_pca))
    println(string("\n\nEKP evaluation ",i," finished. Updating ensemble ...\n"))
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr_pca[j]
    end
    # Get normalized error
    push!(norm_err_list, compute_errors(g_ens_arr, yt_big))
    push!(g_big_list, g_ens_arr)
    if typeof(algo) == Inversion
        update_ensemble!(ekobj, Array(g_ens') , Δt_new=Δt, deterministic_forward_map = deterministic_forward_map)
    else
        update_ensemble!(ekobj, Array(g_ens') )
    end
    println("\nEnsemble updated. Saving results to file...\n")
    # Save EKP information to file
    save(string(outdir_path,"/ekp.jld2"),
        "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
        "ekp_g", get_g(ekobj),
        "truth_mean", ekobj.obs_mean,
        "truth_cov", ekobj.obs_noise_cov,
        "ekp_err", ekobj.err,
        "norm_err", norm_err_list,
        "truth_mean_big", yt_big,
        "g_big", g_big_list,
        "truth_cov_big", yt_var_big,
        "P_pca", P_pca_list,
        "pool_var", pool_var_list,
        )
    # Convert to arrays
    phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
    phi_params_arr = zeros(i+1, n_param, N_ens)
    g_big_arr = zeros(i, N_ens, length(yt_big))
    for (k,elem) in enumerate(phi_params)
      phi_params_arr[k,:,:] = elem
      if k < i + 1
        g_big_arr[k,:,:] = hcat(g_big_list[k]...)'
      end
    end
    norm_err_arr = hcat(norm_err_list...)' # N_iter, N_ens
    npzwrite(string(outdir_path,"/y_mean.npy"), ekobj.obs_mean)
    npzwrite(string(outdir_path,"/Gamma_y.npy"), ekobj.obs_noise_cov)
    npzwrite(string(outdir_path,"/y_mean_big.npy"), yt_big)
    npzwrite(string(outdir_path,"/Gamma_y_big.npy"), yt_var_big)
    npzwrite(string(outdir_path,"/phi_params.npy"), phi_params_arr)
    npzwrite(string(outdir_path,"/norm_err.npy"), norm_err_arr)
    npzwrite(string(outdir_path,"/g_big.npy"), g_big_arr)
    for (l, P_pca) in enumerate(P_pca_list)
      npzwrite(string(outdir_path,"/P_pca_",sim_names[l],".npy"), P_pca)
      npzwrite(string(outdir_path,"/pool_var_",sim_names[l],".npy"), pool_var_list[l])
    end
end

# EKP results: Has the ensemble collapsed toward the truth?
println("\nEKP ensemble mean at last stage (original space):")
println(mean(transform_unconstrained_to_constrained(priors, get_u(ekobj)), dims=1))

println("\nEnsemble covariance det. 1st iteration, transformed space.")
println(det( cov((get_u(ekobj, 1)), dims=1) ))
println("\nEnsemble covariance det. last iteration, transformed space.")
println(det( cov(get_u_final(ekobj), dims=2) ))

