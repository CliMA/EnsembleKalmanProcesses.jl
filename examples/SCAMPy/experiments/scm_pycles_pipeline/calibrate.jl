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
@everywhere using BlockDiagonals
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
                [bounded(0.01, 1.0)]]
# All vars are standard Gaussians in unconstrained space
prior_dist = [Parameterized(Normal(0.0, 1.0))
                for x in range(1, n_param, length=n_param) ]
priors = ParameterDistribution(prior_dist, constraints, param_names)

#########
#########  Define simulation parameters and data directories
#########

# Define observation window (s)
t_starts = [4.0] * 3600  # 4hrs
t_ends = [6.0] * 3600  # 6hrs
# Define variables considered in the loss function
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"])

# Define preconditioning and regularization of inverse problem
perform_PCA = true # Performs PCA on data

# Define directories to fetch data from and store results to
les_names = ["Bomex"]
les_suffixes = ["may18"]
les_root = "/groups/esm/ilopezgo"
scm_names = ["Bomex"]  # same as `les_names` in perfect model setting
scm_data_root = pwd()  # path to folder with `Output.<scm_name>.00000` files
scampy_dir = joinpath(pwd(), "SCAMPy")  # path to SCAMPy
save_full_EDMF_data = false  # if true, save each ensemble output file
outdir_root = pwd()

@assert isdir(les_root) & isdir(scm_data_root) & isdir(scampy_dir) 

#########
#########  Retrieve true LES samples from PyCLES data and transform
#########

# Init arrays
yt = zeros(0)
yt_var_list = Array{Float64, 2}[]
yt_big = zeros(0)
yt_var_list_big = Array{Float64, 2}[]
P_pca_list = []
pool_var_list = []
@assert (  # Each entry in these lists correspond to one simulation case
    length(les_names) == length(les_suffixes) == length(scm_names) 
    == length(y_names) == length(t_starts) == length(t_ends)
)
for (les_name, les_suffix, scm_name, y_name, tstart, tend) in zip(
        les_names, les_suffixes, scm_names, y_names, t_starts, t_ends
    )
    # Get SCM vertical levels for interpolation
    z_scm = get_profile(joinpath(scm_data_root, "Output.$scm_name.00000"), ["z_half"])
    # Get (interpolated and pool-normalized) observations, get pool variance vector
    les_dir = joinpath(les_root, "Output.$les_name.$les_suffix")
    yt_, yt_var_, pool_var = obs_LES(y_name, les_dir, tstart, tend, z_scm = z_scm)
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
    append!(yt_big, yt_)
    push!(yt_var_list_big, yt_var_)
end
d = length(yt) # Length of data array

# Construct global observational covariance matrix, TSVD
Γy = Matrix(BlockDiagonal(yt_var_list))

yt_var_big = cov_from_cov_list(yt_var_list_big)

n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt

#########
#########  Calibrate: Ensemble Kalman Inversion
#########

algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
N_ens = 20 # number of ensemble members
N_iter = 10 # number of EKP iterations.
Δt = 1.0 # Artificial time stepper of the EKI.
println("NUMBER OF ENSEMBLE MEMBERS: $N_ens")
println("NUMBER OF ITERATIONS: $N_iter")

initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo )

# Define caller function
@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(
    x, $param_names, $y_names, $scampy_dir, 
    $scm_data_root, $scm_names, $t_starts, $t_ends,
    P_pca_list = $P_pca_list, norm_var_list = $pool_var_list,
)

# Create output dir
algo_type = typeof(algo) == Sampler{Float64} ? "eks" : "eki"
outdir_path = joinpath(outdir_root, "results_$(algo_type)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$d")
println("Name of outdir path for this EKP is: $outdir_path")
mkpath(outdir_path)

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
    (sim_dirs_arr, g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),3) # Outer dim is G̃, G 
    println(string("\n\nEKP evaluation $i finished. Updating ensemble ...\n"))
    for j in 1:N_ens
        g_ens[j, :] = g_ens_arr_pca[j]
    end
    # Get normalized error for full dimensionality output
    push!(norm_err_list, compute_errors(g_ens_arr, yt_big))
    # Store full dimensionality output
    push!(g_big_list, g_ens_arr)
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
    save(joinpath(outdir_path, "ekp.jld2"),
        "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
        "ekp_g", get_g(ekobj),
        "truth_mean", ekobj.obs_mean,
        "truth_cov", ekobj.obs_noise_cov,
        "ekp_err", ekobj.err,
        "truth_mean_big", yt_big,
        "truth_cov_big", yt_var_big,
        "g_big", g_big_list,
        "norm_err", norm_err_list,
        "P_pca", P_pca_list,
        "pool_var", pool_var_list,
        "phi_params", phi_params_arr,
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
    # Or you can also save information to numpy files with NPZ
    npzwrite(joinpath(outdir_path,"y_mean.npy"), ekobj.obs_mean)
    npzwrite(joinpath(outdir_path,"Gamma_y.npy"), ekobj.obs_noise_cov)
    npzwrite(joinpath(outdir_path,"ekp_err.npy"), ekobj.err)
    npzwrite(joinpath(outdir_path,"phi_params.npy"), phi_params_arr)
    npzwrite(joinpath(outdir_path,"y_mean_big.npy"), yt_big)
    npzwrite(joinpath(outdir_path,"Gamma_y_big.npy"), yt_var_big)
    npzwrite(joinpath(outdir_path,"norm_err.npy"), norm_err_arr)
    npzwrite(joinpath(outdir_path,"g_big.npy"), g_big_arr)
    for (l, P_pca) in enumerate(P_pca_list)
        npzwrite(joinpath(outdir_path,"P_pca_$(scm_names[l]).npy"), P_pca)
        npzwrite(joinpath(outdir_path,"pool_var_$(scm_names[l]).npy"), pool_var_list[l])
    end

    if save_full_EDMF_data
        # Save full EDMF data from every ensemble
        eki_iter_path = joinpath(outdir_path, "EKI_iter_$i")
        mkpath(eki_iter_path)
        # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
        for (ens_i, sim_dirs) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
            ens_i_path = joinpath(eki_iter_path, "ens_$ens_i")
            mkpath(ens_i_path)
            for (scm_name, sim_dir) in zip(scm_names, sim_dirs)
                # Copy simulation data to output directory
                dirname = splitpath(sim_dir)[end]
                @assert dirname[1:7] == "Output."  # sanity check
                # Stats file
                tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
                save_data_path = joinpath(ens_i_path, "Stats.$scm_name.$ens_i.nc")
                run(`cp $tmp_data_path $save_data_path`)
                # namefile and paramfile
                tmp_namefile_path = joinpath(sim_dir, "$scm_name.in")
                save_namefile_path = joinpath(ens_i_path, "$scm_name.in")
                run(`cp $tmp_namefile_path $save_namefile_path`)
                tmp_paramfile_path = joinpath(sim_dir, "paramlist_$scm_name.in")
                save_paramfile_path = joinpath(ens_i_path, "paramlist_$scm_name.in")
                run(`cp $tmp_paramfile_path $save_paramfile_path`)
            end
        end
    end
end
# EKP results: Has the ensemble collapsed toward the truth?
println("\nEKP ensemble mean at last stage (original space):")
println( mean( transform_unconstrained_to_constrained(priors, get_u_final(ekobj)), dims=2) ) # Parameters are stored as columns
