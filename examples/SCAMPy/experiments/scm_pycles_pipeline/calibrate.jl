# This is an example on training the SCAMPy implementation of the EDMF
# scheme with data generated using PyCLES.
#
# The example seeks to find the optimal values of the entrainment and
# detrainment parameters of the EDMF scheme to replicate the LES profiles
# of the BOMEX experiment.
#
# This example is fully parallelized and can be run in the cluster with
# the included script.

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
param_names = ["entrainment_factor", "detrainment_factor"]
n_param = length(param_names)

# Prior information: Define transform to unconstrained gaussian space
constraints = [ [bounded(0.01, 0.3)],
                [bounded(0.01, 0.9)]]
# All vars are standard Gaussians in unconstrained space
prior_dist = [Parameterized(Normal(0.0, 0.5))
                for x in range(1, n_param, length=n_param) ]
priors = ParameterDistribution(prior_dist, constraints, param_names)

#########
#########  Retrieve true LES samples from PyCLES data and transform
#########

# Define observation window (s)
ti = [14400.0]
tf = [21600.0]
# Define variables considered in the loss function
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"])

# Define preconditioning and regularization of inverse problem
perform_PCA = true # Performs PCA on data

sim_names = ["Bomex"]
sim_suffix = [".may18"]

# Init arrays
yt = zeros(0)
yt_var_list = []
P_pca_list = []
pool_var_list = []
for (i, sim_name) in enumerate(sim_names)
    les_dir = string("/groups/esm/ilopezgo/Output.", sim_name, sim_suffix[i])
    # Get SCM vertical levels for interpolation
    z_scm = get_profile(string("Output.", sim_name, ".00000"), ["z_half"])
    # Get (interpolated and pool-normalized) observations, get pool variance vector
    yt_, yt_var_, pool_var = obs_LES(y_names[i], les_dir, ti[i], tf[i], z_scm = z_scm)
    push!(pool_var_list, pool_var)
    if perform_PCA
        yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
        append!(yt, yt_pca)
        push!(yt_var_list, yt_var_pca)
        push!(P_pca_list, P_pca)
    else
        append!(yt, yt_)
        push!(yt_var_list, yt_var_)
        global P_pca_list = nothing
    end
    # Save full dimensionality (normalized) output for error computation
end
d = length(yt) # Length of data array

# Construct global observational covariance matrix, TSVD
yt_var = zeros(d, d)
vars_num = 1
for (k,config_cov) in enumerate(yt_var_list)
    vars = length(config_cov[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = config_cov
    global vars_num = vars_num+vars
end

n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
Γy = yt_var

#########
#########  Calibrate: Ensemble Kalman Inversion
#########

algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
noisy_obs = true
N_ens = 2 # number of ensemble members
N_iter = 1 # number of EKP iterations.
Δt = 1.0
println("NUMBER OF ENSEMBLE MEMBERS: ", N_ens)
println("NUMBER OF ITERATIONS: ", N_iter)

initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo )
scm_dir = "/home/ilopezgo/SCAMPy/"

# Define caller function
@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, $param_names,
   $y_names, $scm_dir, $ti, $tf, P_pca_list = $P_pca_list,
   norm_var_list = $pool_var_list, scampy_handler = "call_BOMEX.sh") 

# Create output dir
prefix = "results_"
prefix = typeof(algo) == Sampler{Float64} ? string(prefix, "eks_") : string(prefix, "eki_")
prefix = Δt ≈ 1 ? prefix : string(prefix, "dt", Δt, "_")
outdir_path = string(prefix, "p", n_param,"_e", N_ens, "_i", N_iter, "_d", d)
println("Name of outdir path for this EKP, ", outdir_path)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# EKP iterations
g_ens = zeros(N_ens, d)
for i in 1:N_iter
    # Note that the parameters are transformed when used as input to SCAMPy
    params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, 
        get_u_final(ekobj)) )
    params = [row[:] for row in eachrow(params_cons_i')]
    @everywhere params = $params
    array_of_tuples = pmap(g_, params) # Outer dim is params iterator
    (g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),2) # Outer dim is G̃, G 
    println(string("\n\nEKP evaluation ",i," finished. Updating ensemble ...\n"))
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr_pca[j]
    end
    # Get normalized error
    if typeof(algo) != Sampler{Float64}
        update_ensemble!(ekobj, Array(g_ens') , Δt_new=Δt)
    else
        update_ensemble!(ekobj, Array(g_ens') )
    end
    println("\nEnsemble updated. Saving results to file...\n")
    # Convert to arrays
    phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
    phi_params_arr = zeros(i+1, n_param, N_ens)
    for (k,elem) in enumerate(phi_params)
      phi_params_arr[k,:,:] = elem
    end

    # Save EKP information to JLD2 file
    save(string(outdir_path,"/ekp.jld2"),
        "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
        "ekp_g", get_g(ekobj),
        "truth_mean", ekobj.obs_mean,
        "truth_cov", ekobj.obs_noise_cov,
        "ekp_err", ekobj.err,
        "P_pca", P_pca_list,
        "pool_var", pool_var_list,
        "phi_params", phi_params_arr,
        )

    # Or you can also save information to numpy files with NPZ
    npzwrite(string(outdir_path,"/y_mean.npy"), ekobj.obs_mean)
    npzwrite(string(outdir_path,"/Gamma_y.npy"), ekobj.obs_noise_cov)
    npzwrite(string(outdir_path,"/ekp_err.npy"), ekobj.err)
    npzwrite(string(outdir_path,"/ekp_g.npy"), get_g(ekobj))
    npzwrite(string(outdir_path,"/phi_params.npy"), phi_params_arr)
    for (l, P_pca) in enumerate(P_pca_list)
      npzwrite(string(outdir_path,"/P_pca_",sim_names[l],".npy"), P_pca)
      npzwrite(string(outdir_path,"/pool_var_",sim_names[l],".npy"), pool_var_list[l])
    end
end

# EKP results: Has the ensemble collapsed toward the truth?
println("\nEKP ensemble mean at last stage (original space):")
println( mean( transform_unconstrained_to_constrained(priors, get_u_final(ekobj)), dims=2) ) # Parameters are stored as columns

