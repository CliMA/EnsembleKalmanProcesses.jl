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
include(joinpath(@__DIR__, "../../src/ekp_plots.jl"))
using JLD2


""" Define parameters and their priors"""
function construct_priors()
    # Define the parameters that we want to learn
    params = Dict(
        # entrainment parameters
        "entrainment_factor"        => [bounded(0.0, 5*0.33)],
        "detrainment_factor"        => [bounded(0.0, 5*0.31)],
    )
    param_names = collect(keys(params))
    constraints = collect(values(params))
    n_param = length(param_names)

    # All vars are approximately uniform in unconstrained space
    prior_dist = repeat([Parameterized(Normal(0.0, 1.78))], n_param)
    priors = ParameterDistribution(prior_dist, constraints, param_names)
    return priors
end

""" Define reference simulations for loss function"""
function construct_reference_models()::Vector{ReferenceModel}
    les_root = "/groups/esm/zhaoyi/pycles_clima"
    scm_root = "/groups/esm/hervik/calibration/static_input"  # path to folder with `Output.<scm_name>.00000` files

    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_bomex = ReferenceModel(
        # Define variables considered in the loss function
        y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        # Reference data specification
        les_root = les_root,
        les_name = "Bomex",
        les_suffix = "aug09",
        # Simulation case specification
        scm_root = scm_root,
        scm_name = "Bomex",
        # Define observation window (s)
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 24.0 * 3600,  # 24hrs
    )
    # Make vector of reference models
    ref_models::Vector{ReferenceModel} = [ref_bomex]
    @assert all(isdir.([les_dir.(ref_models)... scm_dir.(ref_models)...]))

    return ref_models
end

function run_calibrate(return_ekobj=false)
    #########
    #########  Define the parameters and their priors
    #########
    priors = construct_priors()
    

    #########
    #########  Define simulation parameters and data directories
    #########
    ref_models = construct_reference_models()

    outdir_root = pwd()
    # Define preconditioning and regularization of inverse problem
    perform_PCA = false # Performs PCA on data
    normalize = true  # whether to normalize data by pooled variance
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    model_type::Symbol = :les  # :les or :scm
    # Flags for saving output data
    save_eki_data = true  # eki output
    save_ensemble_data = false  # .nc-files from each ensemble run

    #########
    #########  Retrieve true LES samples from PyCLES data and transform
    #########
    
    # Compute data covariance
    ref_stats = ReferenceStatistics(ref_models, model_type, perform_PCA, normalize)
    d = length(ref_stats.y) # Length of data array

    #########
    #########  Calibrate: Ensemble Kalman Inversion
    #########

    algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
    N_ens = 20 # number of ensemble members
    N_iter = 10 # number of EKP iterations.
    Δt = 1.0 # Artificial time stepper of the EKI.
    println("NUMBER OF ENSEMBLE MEMBERS: $N_ens")
    println("NUMBER OF ITERATIONS: $N_iter")

    # parameters are sampled in unconstrained space
    initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=rand(1:1000))
    ekobj = EnsembleKalmanProcess(initial_params, ref_stats.y, ref_stats.Γ, algo )

    # Define caller function
    @everywhere g_(x::Vector{FT}) where FT<:Real = run_SCM(
        x, $priors.names, $ref_models, $ref_stats,
    )

    # Create output dir
    algo_type = typeof(algo) == Sampler{Float64} ? "eks" : "eki"
    n_param = length(priors.names)
    outdir_path = joinpath(outdir_root, "results_$(algo_type)_dt$(Δt)_p$(n_param)_e$(N_ens)_i$(N_iter)_d$(d)_$(model_type)")
    println("Name of outdir path for this EKP is: $outdir_path")
    mkpath(outdir_path)

    # EKP iterations
    g_ens = zeros(N_ens, d)
    norm_err_list = []
    g_big_list = []
    for i in 1:N_iter
        # Parameters are transformed to constrained space when used as input to TurbulenceConvection.jl
        params_cons_i = transform_unconstrained_to_constrained(priors, get_u_final(ekobj))
        params = [c[:] for c in eachcol(params_cons_i)]
        @everywhere params = $params
        array_of_tuples = pmap(g_, params) # Outer dim is params iterator
        (sim_dirs_arr, g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),3) # Outer dim is G̃, G 
        println(string("\n\nEKP evaluation $i finished. Updating ensemble ...\n"))
        for j in 1:N_ens
            if perform_PCA
                g_ens[j, :] = g_ens_arr_pca[j]
            else
                g_ens[j, :] = g_ens_arr[j]
            end
        end

        # Get normalized error
        if typeof(algo) != Sampler{Float64}
            update_ensemble!(ekobj, Array(g_ens') , Δt_new=Δt)
        else
            update_ensemble!(ekobj, Array(g_ens') )
        end
        println("\nEnsemble updated. Saving results to file...\n")

        # Get normalized error for full dimensionality output
        push!(norm_err_list, compute_errors(g_ens_arr, ref_stats.y_full))
        norm_err_arr = hcat(norm_err_list...)' # N_iter, N_ens
        # Store full dimensionality output
        push!(g_big_list, g_ens_arr)
        
        # Convert to arrays
        phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
        phi_params_arr = zeros(i+1, n_param, N_ens)
        g_big_arr = zeros(i, N_ens, full_length(ref_stats))
        for (k,elem) in enumerate(phi_params)
            phi_params_arr[k,:,:] = elem
            if k < i + 1
                g_big_arr[k,:,:] = hcat(g_big_list[k]...)'
            end
        end

        if save_eki_data
            # Save EKP information to JLD2 file
            save(joinpath(outdir_path, "ekp.jld2"),
                "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
                "ekp_g", get_g(ekobj),
                "truth_mean", ekobj.obs_mean,
                "truth_cov", ekobj.obs_noise_cov,
                "ekp_err", ekobj.err,
                "truth_mean_big", ref_stats.y_full,
                "truth_cov_big", ref_stats.Γ_full,
                "P_pca", ref_stats.pca_vec,
                "pool_var", ref_stats.norm_vec,
                "g_big", g_big_list,
                "g_big_arr", g_big_arr,
                "norm_err", norm_err_list,
                "norm_err_arr", norm_err_arr,
                "phi_params", phi_params_arr,
            )

            # make ekp plots
            make_ekp_plots(outdir_path, priors.names)
        end

        
        if save_ensemble_data
            eki_iter_path = joinpath(outdir_path, "EKI_iter_$i")
            mkpath(eki_iter_path)
            save_full_ensemble_data(eki_iter_path, sim_dirs_arr, scm_names)
        end
    end
    # EKP results: Has the ensemble collapsed toward the truth?
    println("\nEKP ensemble mean at last stage (original space):")
    println( mean( transform_unconstrained_to_constrained(priors, get_u_final(ekobj)), dims=2) ) # Parameters are stored as columns

    if return_ekobj
        return ekobj, outdir_path
    end
end


""" Save full EDMF data from every ensemble"""
function save_full_ensemble_data(save_path, sim_dirs_arr, scm_names)
    # get a simulation directory `.../Output.SimName.UUID`, and corresponding parameter name
    for (ens_i, sim_dirs) in enumerate(sim_dirs_arr)  # each ensemble returns a list of simulation directories
        ens_i_path = joinpath(save_path, "ens_$ens_i")
        mkpath(ens_i_path)
        for (scm_name, sim_dir) in zip(scm_names, sim_dirs)
            # Copy simulation data to output directory
            dirname = splitpath(sim_dir)[end]
            @assert dirname[1:7] == "Output."  # sanity check
            # Stats file
            tmp_data_path = joinpath(sim_dir, "stats/Stats.$scm_name.nc")
            save_data_path = joinpath(ens_i_path, "Stats.$scm_name.$ens_i.nc")
            cp(tmp_data_path, save_data_path)
            # namefile
            tmp_namefile_path = namelist_directory(sim_dir, scm_name)
            save_namefile_path = namelist_directory(ens_i_path, scm_name)
            cp(tmp_namefile_path, save_namefile_path)
        end
    end
end
