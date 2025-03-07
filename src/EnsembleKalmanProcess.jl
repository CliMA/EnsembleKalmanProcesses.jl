using ..ParameterDistributions
using ..DataContainers
using ..Localizers

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EnsembleKalmanProcess
export get_u, get_g, get_ϕ
export get_u_prior, get_u_final, get_g_final, get_ϕ_final
export get_N_iterations, get_error, get_cov_blocks
export get_u_mean, get_u_cov, get_g_mean, get_ϕ_mean
export get_u_mean_final, get_u_cov_prior, get_u_cov_final, get_g_mean_final, get_ϕ_mean_final
export get_scheduler,
    get_localizer, get_localizer_type, get_accelerator, get_rng, get_Δt, get_failure_handler, get_N_ens, get_process
export get_observation_series, get_obs, get_obs_noise_cov, get_obs_noise_cov_inv
export compute_error!
export update_ensemble!
export list_update_groups_over_minibatch
export sample_empirical_gaussian, split_indices_by_success
export SampleSuccGauss, IgnoreFailures, FailureHandler


abstract type Process end
#specific Processes and their exports are included after the general definitions

# LearningRateSchedulers
abstract type LearningRateScheduler end

# Failure handlers
abstract type FailureHandlingMethod end

# Accelerators
abstract type Accelerator end



"Failure handling method that ignores forward model failures"
struct IgnoreFailures <: FailureHandlingMethod end

""""
    SampleSuccGauss <: FailureHandlingMethod

Failure handling method that substitutes failed ensemble members by new samples from
the empirical Gaussian distribution defined by the updated successful ensemble.
"""
struct SampleSuccGauss <: FailureHandlingMethod end

"""
    FailureHandler{P <: Process, FM <: FailureHandlingMethod}

Structure defining the failure handler method used in the EnsembleKalmanProcess.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct FailureHandler{P <: Process, FM <: FailureHandlingMethod}
    "Failsafe algorithmic update equation"
    failsafe_update::Function
end


function default_options_dict(process::P) where {P <: Process}
    if isa(process, Inversion)
        return Dict(
            "scheduler" => DataMisfitController(terminate_at = 1),
            "localization_method" => SECNice(),
            "failure_handler_method" => SampleSuccGauss(),
            "accelerator" => NesterovAccelerator(),
        )
    elseif isa(process, TransformInversion)
        return Dict(
            "scheduler" => DataMisfitController(terminate_at = 1),
            "localization_method" => NoLocalization(),
            "failure_handler_method" => SampleSuccGauss(),
            "accelerator" => DefaultAccelerator(),
        )
    elseif isa(process, Sampler)
        return Dict(
            "scheduler" => EKSStableScheduler(1.0, eps()),
            "localization_method" => NoLocalization(),
            "failure_handler_method" => IgnoreFailures(),
            "accelerator" => DefaultAccelerator(),
        )
    elseif isa(process, Unscented)
        return Dict(
            "scheduler" => DataMisfitController(terminate_at = 1),
            "localization_method" => NoLocalization(),
            "failure_handler_method" => SampleSuccGauss(),
            "accelerator" => DefaultAccelerator(),
        )
    elseif isa(process, SparseInversion)
        return Dict(
            "scheduler" => DefaultScheduler(),
            "localization_method" => SECNice(),
            "failure_handler_method" => SampleSuccGauss(),
            "accelerator" => DefaultAccelerator(),
        )
    elseif isa(process, GaussNewtonInversion)
        return Dict(
            "scheduler" => DefaultScheduler(),
            "localization_method" => SECNice(),
            "failure_handler_method" => SampleSuccGauss(),
            "accelerator" => NesterovAccelerator(),
        )
    else
        throw(
            ArgumentError(
                "No defaults found for process $process, please implement these in EnsembleKalmanProcess.jl default_options_dict()",
            ),
        )
    end

end

## begin general constructor and function definitions

"""
    EnsembleKalmanProcess{FT <: AbstractFloat, IT <: Int, P <: Process}

Structure that is used in Ensemble Kalman processes.

# Fields

$(TYPEDFIELDS)

# Generic constructor

    EnsembleKalmanProcess(
        params::AbstractMatrix{FT},
        observation_series::OS,
        obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
        process::P;
        scheduler = DefaultScheduler(1),
        Δt = FT(1),
        rng::AbstractRNG = Random.GLOBAL_RNG,
        failure_handler_method::FM = IgnoreFailures(),
        localization_method::LM = NoLocalization(),
        verbose::Bool = false,
    ) where {FT <: AbstractFloat, P <: Process, FM <: FailureHandlingMethod, LM <: LocalizationMethod, OS <: ObservationSeries}

Inputs:

 - `params`                 :: Initial parameter ensemble
 - `observation_series`     :: Container for observations (and possible minibatching)
 - `process`                :: Algorithm used to evolve the ensemble
 - `scheduler`              :: Adaptive timestep calculator 
 - `Δt`                     :: Initial time step or learning rate
 - `rng`                    :: Random number generator
 - `failure_handler_method` :: Method used to handle particle failures
 - `localization_method`    :: Method used to localize sample covariances
 - `verbose`                :: Whether to print diagnostic information

# Other constructors:

$(METHODLIST)
"""
struct EnsembleKalmanProcess{
    FT <: AbstractFloat,
    IT <: Int,
    P <: Process,
    LRS <: LearningRateScheduler,
    ACC <: Accelerator,
    VV <: AbstractVector,
}
    "array of stores for parameters (`u`), each of size [`N_par × N_ens`]"
    u::Array{DataContainer{FT}}
    "Container for the observation(s) - and minibatching mechanism"
    observation_series::ObservationSeries
    "ensemble size"
    N_ens::IT
    "Array of stores for forward model outputs, each of size  [`N_obs × N_ens`]"
    g::Array{DataContainer{FT}}
    "vector of errors"
    error::Vector{FT}
    "Scheduler to calculate the timestep size in each EK iteration"
    scheduler::LRS
    "accelerator object that informs EK update steps, stores additional state variables as needed"
    accelerator::ACC
    "stored vector of timesteps used in each EK iteration"
    Δt::Vector{FT}
    "vector of update groups, defining which parameters should be updated by which data"
    update_groups::VV
    "the particular EK process (`Inversion` or `Sampler` or `Unscented` or `TransformInversion` or `SparseInversion`)"
    process::P
    "Random number generator object (algorithm + seed) used for sampling and noise, for reproducibility. Defaults to `Random.GLOBAL_RNG`."
    rng::AbstractRNG
    "struct storing failsafe update directives, implemented for (`Inversion`, `SparseInversion`, `Unscented`, `TransformInversion`)"
    failure_handler::FailureHandler
    "Localization kernel, implemented for (`Inversion`, `SparseInversion`, `Unscented`)"
    localizer::Localizer
    "Whether to print diagnostics for each EK iteration"
    verbose::Bool
end

# outer constructors
function EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    observation_series::OS,
    process::P,
    configuration::Dict;
    update_groups::Union{Nothing, VV} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = false,
) where {FT <: AbstractFloat, P <: Process, OS <: ObservationSeries, VV <: AbstractVector}

    #initial parameters stored as columns
    init_params = DataContainer(params, data_are_columns = true)

    # dimensionality
    N_par, N_ens = size(init_params) #stored with data as columns

    if N_ens < 10
        @warn "Recommended minimum ensemble size (`N_ens`) is 10. Got `N_ens` = $(N_ens)."
    elseif (N_par < 10) && (N_ens < 10 * N_par)
        @warn "For $(N_par) parameters, the recommended minimum ensemble size (`N_ens`) is $(10*(N_par)). Got `N_ens` = $(N_ens)`."
    end
    if (N_par >= 10) && (N_ens < 100)
        @warn "For $(N_par) parameters, the recommended minimum ensemble size (`N_ens`) is 100. Got `N_ens` = $(N_ens)`."
    end

    obs_for_minibatch = get_obs(observation_series) # get stacked observation over minibatch
    obs_size_for_minibatch = length(obs_for_minibatch) # number of dims in the stacked observation
    IT = typeof(N_ens)
    #store for model evaluations
    g = []
    # error store
    err = FT[]
    # timestep store
    Δt = FT[]

    # defined groups of parameters to be updated by groups of data 
    obs_size = length(get_obs(get_observations(observation_series)[1])) #deduce size just from first observation
    if isnothing(update_groups)
        groups = [UpdateGroup(1:N_par, 1:obs_size)] # vec length 1
    else
        groups = update_groups
    end
    update_group_consistency(groups, N_par, obs_size) # consistency checks
    VVV = typeof(groups)

    scheduler = configuration["scheduler"]
    RS = typeof(scheduler)

    # set up accelerator
    accelerator = configuration["accelerator"]
    AC = typeof(accelerator)
    if !(isa(accelerator, DefaultAccelerator))
        set_ICs!(accelerator, params)
        if isa(process, Sampler)
            @warn "Acceleration is experimental for Sampler processes and may affect convergence."
        end
    end

    # failure handler
    fh_method = configuration["failure_handler_method"]
    failure_handler = FailureHandler(process, fh_method)

    # localizer
    loc_method = configuration["localization_method"]
    if isa(process, TransformInversion) && !(isa(loc_method, NoLocalization))
        throw(ArgumentError("`TransformInversion` cannot currently be used with localization."))
    end

    localizer = Localizer(loc_method, N_ens, FT)

    if verbose
        @info "Initializing ensemble Kalman process of type $(nameof(typeof(process)))\nNumber of ensemble members: $(N_ens)\nLocalization: $(nameof(typeof(loc_method)))\nFailure handler: $(nameof(typeof(fh_method)))\nScheduler: $(nameof(typeof(scheduler)))\nAccelerator: $(nameof(typeof(accelerator)))"
    end

    EnsembleKalmanProcess{FT, IT, P, RS, AC, VVV}(
        [init_params],
        observation_series,
        N_ens,
        g,
        err,
        scheduler,
        accelerator,
        Δt,
        groups,
        process,
        rng,
        failure_handler,
        localizer,
        verbose,
    )
end

function EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    observation_series::OS,
    process::P;
    scheduler::Union{Nothing, LRS} = nothing,
    accelerator::Union{Nothing, ACC} = nothing,
    failure_handler_method::Union{Nothing, FM} = nothing,
    localization_method::Union{Nothing, LM} = nothing,
    Δt = nothing,
    update_groups::Union{Nothing, VV} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    verbose::Bool = false,
) where {
    FT <: AbstractFloat,
    VV <: AbstractVector,
    LRS <: LearningRateScheduler,
    ACC <: Accelerator,
    P <: Process,
    FM <: FailureHandlingMethod,
    LM <: LocalizationMethod,
    OS <: ObservationSeries,
}

    if !(isnothing(Δt))
        @warn "the `Δt = x` keyword argument is deprecated, ignoring... for the same behavior please set `scheduler = DefaultScheduler(x)`, or `scheduler = EKSStableScheduler()` for using the `Sampler` "
    end

    # get defaults for scheduler, accelerator, failure handling, localization
    configuration = default_options_dict(process)
    # override if necessary
    if !isnothing(scheduler)
        configuration["scheduler"] = scheduler
    end
    if !isnothing(accelerator)
        configuration["accelerator"] = accelerator
    end
    if !isnothing(failure_handler_method)
        configuration["failure_handler_method"] = failure_handler_method
    end
    if !isnothing(localization_method)
        configuration["localization_method"] = localization_method
    end

    return EnsembleKalmanProcess(
        params,
        observation_series,
        process,
        configuration,
        update_groups = update_groups,
        rng = rng,
        verbose = verbose,
    )
end

function EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    observation::OB,
    args...;
    kwargs...,
) where {FT <: AbstractFloat, OB <: Observation}
    observation_series = ObservationSeries(observation)
    return EnsembleKalmanProcess(params, observation_series, args...; kwargs...)
end

function EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    obs,
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    args...;
    kwargs...,
) where {FT <: AbstractFloat}

    observation = Observation(Dict("samples" => obs, "covariances" => obs_noise_cov, "names" => "observation"))

    return EnsembleKalmanProcess(params, observation, args...; kwargs...)
end

include("LearningRateSchedulers.jl")

"""
    get_u(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}

Returns the unconstrained parameters at the given iteration. Returns a DataContainer object unless `return_array` is true.
"""
function get_u(ekp::EnsembleKalmanProcess, iteration::IT; return_array = true) where {IT <: Integer}
    return return_array ? get_data(ekp.u[iteration]) : ekp.u[iteration]
end

"""
    get_g(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}

Returns the forward model evaluations at the given iteration. Returns a `DataContainer` object unless `return_array` is true.
"""
function get_g(ekp::EnsembleKalmanProcess, iteration::IT; return_array = true) where {IT <: Integer}
    return return_array ? get_data(ekp.g[iteration]) : ekp.g[iteration]
end

"""
    get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess, iteration::IT; return_array=true)

Returns the constrained parameters at the given iteration.
"""
function get_ϕ(
    prior::ParameterDistribution,
    ekp::EnsembleKalmanProcess,
    iteration::IT;
    return_array = true,
) where {IT <: Integer}
    return transform_unconstrained_to_constrained(prior, get_u(ekp, iteration, return_array = return_array))
end

"""
    get_u(ekp::EnsembleKalmanProcess; return_array=true)

Returns the unconstrained parameters from all iterations. The outer dimension is given by the number of iterations,
and the inner objects are `DataContainer` objects unless `return_array` is true.
"""
function get_u(ekp::EnsembleKalmanProcess; return_array = true)
    N_stored_u = get_N_iterations(ekp) + 1
    return [get_u(ekp, it, return_array = return_array) for it in 1:N_stored_u]
end

"""
    get_g(ekp::EnsembleKalmanProcess; return_array=true)

Returns the forward model evaluations from all iterations. The outer dimension is given by the number of iterations,
and the inner objects are `DataContainer` objects unless `return_array` is true.
"""
function get_g(ekp::EnsembleKalmanProcess; return_array = true)
    N_stored_g = get_N_iterations(ekp)
    return [get_g(ekp, it, return_array = return_array) for it in 1:N_stored_g]
end

"""
    get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess; return_array=true)

Returns the constrained parameters from all iterations. The outer dimension is given by the number of iterations,
and the inner objects are `DataContainer` objects unless `return_array` is true.
"""
get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess; return_array = true) =
    transform_unconstrained_to_constrained(prior, get_u(ekp, return_array = return_array))

"""
    get_u_mean(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}

Returns the mean unconstrained parameter at the given iteration.
"""
function get_u_mean(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}
    return vec(mean(get_data(ekp.u[iteration]), dims = 2))
end

"""
    get_ϕ_mean(prior::ParameterDistribution, ekp::EnsembleKalmanProcess, iteration::IT)

Returns the constrained transform of the mean unconstrained parameter at the given iteration.
"""
function get_ϕ_mean(prior::ParameterDistribution, ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}
    return transform_unconstrained_to_constrained(prior, get_u_mean(ekp, iteration))
end

"""
    get_u_cov(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}

Returns the unconstrained parameter sample covariance at the given iteration.
"""
function get_u_cov(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}
    u = get_data(ekp.u[iteration])
    return cov(u, dims = 2)
end

"""
    get_u_cov_prior(ekp::EnsembleKalmanProcess)

Returns the unconstrained parameter sample covariance for the initial ensemble.
"""
function get_u_cov_prior(ekp::EnsembleKalmanProcess)
    return cov(get_u_prior(ekp), dims = 2)
end

"""
    get_g_mean(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}

Returns the mean forward map evaluation at the given iteration.
"""
function get_g_mean(ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}
    return vec(mean(get_data(ekp.g[iteration]), dims = 2))
end

"""
    get_u_final(ekp::EnsembleKalmanProcess; return_array=true)

Get the unconstrained parameters at the last iteration, returning a `DataContainer` Object if `return_array` is false.
"""
function get_u_final(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_u(ekp, size(ekp.u, 1)) : ekp.u[end]
end

"""
    get_u_prior(ekp::EnsembleKalmanProcess; return_array=true)

Get the unconstrained parameters as drawn from the prior, returning a `DataContainer` Object if `return_array` is false.
"""
function get_u_prior(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_u(ekp, 1) : ekp.u[1]
end

"""
    get_g_final(ekp::EnsembleKalmanProcess; return_array=true)

Get forward model outputs at the last iteration, returns a `DataContainer` Object if `return_array` is false.
"""
function get_g_final(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_g(ekp, size(ekp.g, 1)) : ekp.g[end]
end

"""
    get_ϕ_final(ekp::EnsembleKalmanProcess; return_array=true)

Get the constrained parameters at the last iteration.
"""
get_ϕ_final(prior::ParameterDistribution, ekp::EnsembleKalmanProcess; return_array = true) =
    transform_unconstrained_to_constrained(prior, get_u_final(ekp, return_array = return_array))

"""
    get_u_mean_final(ekp::EnsembleKalmanProcess)

Get the mean unconstrained parameter at the last iteration.
"""
get_u_mean_final(ekp::EnsembleKalmanProcess) = get_u_mean(ekp, size(ekp.u, 1))

"""
    get_ϕ_mean_final(prior::ParameterDistribution, ekp::EnsembleKalmanProcess)

Get the constrained transform of the mean unconstrained parameter at the last iteration.
"""
get_ϕ_mean_final(prior::ParameterDistribution, ekp::EnsembleKalmanProcess) =
    transform_unconstrained_to_constrained(prior, get_u_mean_final(ekp))

"""
    get_u_cov_final(ekp::EnsembleKalmanProcess)

Get the mean unconstrained parameter covariance at the last iteration.
"""
get_u_cov_final(ekp::EnsembleKalmanProcess) = get_u_cov(ekp, size(ekp.u, 1))

"""
    get_g_mean_final(ekp::EnsembleKalmanProcess)

Get the mean forward model evaluation at the last iteration.
"""
get_g_mean_final(ekp::EnsembleKalmanProcess) = get_g_mean(ekp, size(ekp.g, 1))

"""
    get_N_iterations(ekp::EnsembleKalmanProcess)

Get number of times update has been called (equals `size(g)`, or `size(u)-1`).
"""
function get_N_iterations(ekp::EnsembleKalmanProcess)
    return size(ekp.u, 1) - 1
end

# basic getters
"""
    get_N_ens(ekp::EnsembleKalmanProcess)
Return `N_ens` field of EnsembleKalmanProcess.
"""
function get_N_ens(ekp::EnsembleKalmanProcess)
    return ekp.N_ens
end

"""
    get_Δt(ekp::EnsembleKalmanProcess)
Return `Δt` field of EnsembleKalmanProcess.
"""
function get_Δt(ekp::EnsembleKalmanProcess)
    return ekp.Δt
end

"""
    get_failuer_handler(ekp::EnsembleKalmanProcess)
Return `failure_handler` field of EnsembleKalmanProcess.
"""
function get_failure_handler(ekp::EnsembleKalmanProcess)
    return ekp.failure_handler
end

"""
    get_update_groups(ekp::EnsembleKalmanProcess)
Return update_groups type of EnsembleKalmanProcess.
"""
function get_update_groups(ekp::EnsembleKalmanProcess)
    return ekp.update_groups
end

"""
    list_update_groups_over_minibatch(ekp::EnsembleKalmanProcess)
Return u_groups and g_groups for the current minibatch, i.e. the subset of 
"""
function list_update_groups_over_minibatch(ekp::EnsembleKalmanProcess)
    os = get_observation_series(ekp)
    len_mb = length(get_current_minibatch(os)) # number of obs per batch
    len_obs = Int(length(get_obs(os)) / len_mb) # length of obs in a batch
    update_groups = get_update_groups(ekp)
    u_groups = get_u_group.(update_groups) # update_group indices
    g_groups = get_g_group.(update_groups)
    # extend group indices from one obs to the minibatch of obs
    new_u_groups = u_groups
    new_g_groups = [reduce(vcat, [(i - 1) * len_obs .+ g_group for i in 1:len_mb]) for g_group in g_groups]

    return new_u_groups, new_g_groups
end




"""
    get_process(ekp::EnsembleKalmanProcess)
Return `process` field of EnsembleKalmanProcess.
"""
function get_process(ekp::EnsembleKalmanProcess)
    return ekp.process
end

"""
    get_localizer(ekp::EnsembleKalmanProcess)
Return `localizer` field of EnsembleKalmanProcess.
"""
function get_localizer(ekp::EnsembleKalmanProcess)
    return ekp.localizer
end

"""
    get_localizer_type(ekp::EnsembleKalmanProcess)
Return first parametric type of the `localizer` field of EnsembleKalmanProcess.
"""
function get_localizer_type(ekp::EnsembleKalmanProcess)
    return Localizers.get_localizer(get_localizer(ekp))
end

"""
    get_scheduler(ekp::EnsembleKalmanProcess)
Return `scheduler` field of EnsembleKalmanProcess.
"""
function get_scheduler(ekp::EnsembleKalmanProcess)
    return ekp.scheduler
end

"""
    get_accelerator(ekp::EnsembleKalmanProcess)
Return `accelerator` field of EnsembleKalmanProcess.
"""
function get_accelerator(ekp::EnsembleKalmanProcess)
    return ekp.accelerator
end

"""
    get_rng(ekp::EnsembleKalmanProcess)
Return `rng` field of EnsembleKalmanProcess.
"""
function get_rng(ekp::EnsembleKalmanProcess)
    return ekp.rng
end

"""
    get_observation_series(ekp::EnsembleKalmanProcess)
Return `observation_series` field of EnsembleKalmanProcess.
"""
function get_observation_series(ekp::EnsembleKalmanProcess)
    return ekp.observation_series
end

"""
    get_obs_noise_cov(ekp::EnsembleKalmanProcess; build=true)
convenience function to get the obs_noise_cov from the current batch in ObservationSeries
build=false:, returns a vector of blocks,
build=true: returns a block matrix,
"""
function get_obs_noise_cov(ekp::EnsembleKalmanProcess; build = true)
    return get_obs_noise_cov(get_observation_series(ekp), build = build)
end

"""
    get_obs_noise_cov_inv(ekp::EnsembleKalmanProcess; build=true)
convenience function to get the obs_noise_cov (inverse) from the current batch in ObservationSeries
build=false:, returns a vector of blocks,
build=true: returns a block matrix,
"""
function get_obs_noise_cov_inv(ekp::EnsembleKalmanProcess; build = true)
    return get_obs_noise_cov_inv(get_observation_series(ekp), build = build)
end

"""
    get_obs(ekp::EnsembleKalmanProcess; build=true)
Get the observation from the current batch in ObservationSeries
build=false: returns a vector of vectors,
build=true: returns a concatenated vector,
"""
function get_obs(ekp::EnsembleKalmanProcess; build = true)
    return get_obs(get_observation_series(ekp), build = build)
end

"""
    update_minibatch!(ekp::EnsembleKalmanProcess)
update to the next minibatch in the ObservationSeries
"""
function update_minibatch!(ekp::EnsembleKalmanProcess)
    return update_minibatch!(get_observation_series(ekp))
end

function get_current_minibatch(ekp::EnsembleKalmanProcess)
    return get_current_minibatch(get_observation_series(ekp))
end

"""
    construct_initial_ensemble(
        rng::AbstractRNG,
        prior::ParameterDistribution,
        N_ens::IT
    ) where {IT <: Int}
    construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT) where {IT <: Int}

Construct the initial parameters, by sampling `N_ens` samples from specified
prior distribution. Returned with parameters as columns.
"""
function construct_initial_ensemble(rng::AbstractRNG, prior::ParameterDistribution, N_ens::IT) where {IT <: Int}
    return sample(rng, prior, N_ens) #of size [dim(param space) N_ens]
end
# first arg optional; defaults to GLOBAL_RNG (as in Random, StatsBase)
construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT) where {IT <: Int} =
    construct_initial_ensemble(Random.GLOBAL_RNG, prior, N_ens)

"""
    compute_error!(ekp::EnsembleKalmanProcess)

Computes the covariance-weighted error of the mean forward model output, `(ḡ - y)'Γ_inv(ḡ - y)`.
The error is stored within the `EnsembleKalmanProcess`.
"""
function compute_error!(ekp::EnsembleKalmanProcess)
    mean_g = dropdims(mean(get_g_final(ekp), dims = 2), dims = 2)
    diff = get_obs(ekp) - mean_g
    Γ_inv = get_obs_noise_cov_inv(ekp, build = false)
    γ_sizes = [size(γ_inv, 1) for γ_inv in Γ_inv]
    X = zeros(sum(γ_sizes), size(diff, 2)) # stores Y' * Γ_inv
    shift = [0]
    for (γs, γ_inv) in zip(γ_sizes, Γ_inv)
        idx = (shift[1] + 1):(shift[1] + γs)
        X[idx, :] = γ_inv * diff[idx, :]
        shift[1] = maximum(idx)
    end
    newerr = dot(diff, X)
    push!(get_error(ekp), newerr)
end

"""
    get_error(ekp::EnsembleKalmanProcess)

Returns the mean forward model output error as a function of algorithmic time.
"""
get_error(ekp::EnsembleKalmanProcess) = ekp.error


"""
    sample_empirical_gaussian(
        rng::AbstractRNG,
        u::AbstractMatrix{FT},
        n::IT;
        inflation::Union{FT, Nothing} = nothing,
    ) where {FT <: Real, IT <: Int}

Returns `n` samples from an empirical Gaussian based on point estimates `u`, adding inflation if the covariance is singular.
"""
function sample_empirical_gaussian(
    rng::AbstractRNG,
    u::AbstractMatrix{FT},
    n::IT;
    inflation::Union{FT, Nothing} = nothing,
) where {FT <: Real, IT <: Int}
    cov_u_new = Symmetric(cov(u, dims = 2))
    if !isposdef(cov_u_new)
        @warn string("Sample covariance matrix over ensemble is singular.", "\n Applying variance inflation.")
        if isnothing(inflation)
            # Reduce condition number to 1/sqrt(eps(FT))
            inflation = eigmax(cov_u_new) * sqrt(eps(FT))
        end
        cov_u_new = cov_u_new + inflation * I
    end
    mean_u_new = mean(u, dims = 2)
    return mean_u_new .+ sqrt(cov_u_new) * rand(rng, MvNormal(zeros(length(mean_u_new[:])), I), n)
end

function sample_empirical_gaussian(
    u::AbstractMatrix{FT},
    n::IT;
    inflation::Union{FT, Nothing} = nothing,
) where {FT <: Real, IT <: Int}
    return sample_empirical_gaussian(Random.GLOBAL_RNG, u, n, inflation = inflation)
end


"""
     split_indices_by_success(g::AbstractMatrix{FT}) where {FT <: Real}

Returns the successful/failed particle indices given a matrix with output vectors stored as columns.
Failures are defined for particles containing at least one NaN output element.
"""
function split_indices_by_success(g::AbstractMatrix{FT}) where {FT <: Real}
    failed_ens = [i for i = 1:size(g, 2) if any(isnan.(g[:, i]))]
    successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
    if length(failed_ens) > length(successful_ens)
        @warn string(
            "More than 50% of runs produced NaNs ($(length(failed_ens))/$(size(g, 2))).",
            "\nIterating... but consider increasing model stability.",
            "\nThis will affect optimization result.",
        )
    end
    return successful_ens, failed_ens
end

"""
    get_cov_blocks(cov::AbstractMatrix{FT}, p::IT) where {FT <: Real, IT <: Integer}

Given a covariance matrix `cov` and number of parameters `p`, returns the matrix blocks corresponding to the u–u
covariance, the u–G(u) covariance, and the G(u)–G(u) covariance.
"""
function get_cov_blocks(cov::AbstractMatrix{FT}, p::IT) where {FT <: Real, IT <: Integer}
    uu_cov = cov[1:p, 1:p]
    ug_cov = cov[1:p, (p + 1):end]
    gg_cov = cov[(p + 1):end, (p + 1):end]
    return uu_cov, ug_cov, gg_cov
end

"""
$(TYPEDSIGNATURES)

Applies multiplicative noise to particles, and is aware of the current Δt (see Docs page for details). 
Inputs:
    - ekp :: The EnsembleKalmanProcess to update.
    - s :: Scaling factor for time step in multiplicative perturbation.
"""
function multiplicative_inflation!(ekp::EnsembleKalmanProcess; s::FT = 1.0) where {FT <: Real}

    scaled_Δt = s * get_Δt(ekp)[end]

    if scaled_Δt >= 1.0
        error(string("Scaled time step: ", scaled_Δt, " is >= 1.0", "\nChange s or EK time step."))
    end

    u = get_u_final(ekp)
    u_mean = get_u_mean_final(ekp)
    prefactor = sqrt(1 / (1 - scaled_Δt))
    u_updated = u_mean .+ prefactor * (u .- u_mean)
    ekp.u[end] = DataContainer(u_updated, data_are_columns = true)

end

"""
$(TYPEDSIGNATURES)

Applies additive Gaussian noise to particles. Noise is drawn from normal distribution with 0 mean
and scaled parameter covariance, and accounting for the current Δt . The original parameter covariance is a provided matrix, assumed positive semi-definite.
Inputs:
    - ekp :: The EnsembleKalmanProcess to update.
    - s :: Scaling factor for time step in additive perturbation.
    - inflation_cov :: AbstractMatrix provide a N_par x N_par matrix to use.
"""
function additive_inflation!(
    ekp::EnsembleKalmanProcess,
    inflation_cov::MorUS;
    s::FT = 1.0,
) where {FT <: Real, MorUS <: Union{AbstractMatrix, UniformScaling}}

    scaled_Δt = s * get_Δt(ekp)[end]

    if scaled_Δt >= 1.0
        error(string("Scaled time step: ", scaled_Δt, " is >= 1.0", "\nChange s or EK time step."))
    end

    u = get_u_final(ekp)

    Σ_sqrt = sqrt(scaled_Δt / (1 - scaled_Δt) .* inflation_cov)

    # add multivariate noise with 0 mean and scaled covariance
    u_updated = u .+ Σ_sqrt * rand(get_rng(ekp), MvNormal(zeros(size(u, 1)), I), size(u, 2))
    ekp.u[end] = DataContainer(u_updated, data_are_columns = true)
end




"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess,
        g::AbstractMatrix{FT};
        multiplicative_inflation::Bool = false,
        additive_inflation::Bool = false,
        additive_inflation_cov::MorUS = get_u_cov_prior(ekp),
        s::FT = 0.0,
        ekp_kwargs...,
    ) where {FT, IT}
Updates the ensemble according to an Inversion process.
Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - multiplicative_inflation :: Flag indicating whether to use multiplicative inflation.
 - additive_inflation :: Flag indicating whether to use additive inflation.
 - additive_inflation_cov ::  specifying an additive inflation matrix (default is the prior covariance) assumed positive semi-definite
        If false (default), parameter covariance from the current iteration is used.
 - s :: Scaling factor for time step in inflation step.
 - ekp_kwargs :: Keyword arguments to pass to standard ekp update_ensemble!.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess,
    g::AbstractMatrix{FT};
    multiplicative_inflation::Bool = false,
    additive_inflation::Bool = false,
    additive_inflation_cov::MorUS = get_u_cov_prior(ekp),
    s::FT = 0.0,
    Δt_new::NFT = nothing,
    ekp_kwargs...,
) where {FT, NFT <: Union{Nothing, AbstractFloat}, MorUS <: Union{AbstractMatrix, UniformScaling}}
    #catch works when g non-square 
    if !(size(g)[2] == get_N_ens(ekp))
        throw(
            DimensionMismatch(
                "ensemble size $(get_N_ens(ekp)) in EnsembleKalmanProcess does not match the columns of g ($(size(g)[2])); try transposing g or check the ensemble size",
            ),
        )
    end
    # check if columns of g are the same (and not NaN)
    n_nans = sum(isnan.(sum(g, dims = 1)))
    nan_adjust = (n_nans > 0) ? -n_nans + 1 : 0
    # as unique reduces NaNs to one column if present. or 0 if not
    if length(unique(eachcol(g))) < size(g, 2) + nan_adjust
        nonunique_cols = size(g, 2) + nan_adjust - length(unique(eachcol(g)))
        @warn "Detected $(nonunique_cols) clashes where forward map evaluations are exactly equal (and not NaN), this is likely to cause `LinearAlgebra` difficulty. Please check forward evaluations for bugs."
    end

    terminate = calculate_timestep!(ekp, g, Δt_new)
    if isnothing(terminate)

        if ekp.verbose
            cov_init = get_u_cov_final(ekp)
            if get_N_iterations(ekp) == 0
                @info "Iteration 0 (prior)"
                @info "Covariance trace: $(tr(cov_init))"
            end

            @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(get_Δt(ekp))))"
        end

        u_groups, g_groups = list_update_groups_over_minibatch(ekp)
        u = zeros(size(get_u_prior(ekp)))

        # update each u_block with every g_block
        for (u_idx, g_idx) in zip(u_groups, g_groups)
            u[u_idx, :] += update_ensemble!(ekp, g, get_process(ekp), u_idx, g_idx; ekp_kwargs...)
        end

        accelerate!(ekp, u)

        if s > 0.0 #if user specifies inflation 
            multiplicative_inflation ? multiplicative_inflation!(ekp; s = s) : nothing
            additive_inflation ? additive_inflation!(ekp, additive_inflation_cov, s = s) : nothing
        else # if, by default there is inflation due to the process imposing the prior
            process = get_process(ekp)
            if any([isa(process, Inversion), isa(process, TransformInversion)])
                if get_impose_prior(process) # if true then add inflation
                    # need sΔt < 1 
                    ss = get_default_multiplicative_inflation(process) * min(1.0, 1.0 / get_Δt(ekp)[end]) # heuristic to bound ss for very large timesteps.
                    multiplicative_inflation!(ekp; s = ss)
                end
            end

        end


        # wrapping up
        push!(ekp.g, DataContainer(g, data_are_columns = true)) # store g
        compute_error!(ekp)

        if ekp.verbose
            cov_new = get_u_cov_final(ekp)
            @info "Covariance-weighted error: $(get_error(ekp)[end])\nCovariance trace: $(tr(cov_new))\nCovariance trace ratio (current/previous): $(tr(cov_new)/tr(cov_init))"
        end

    else
        return terminate # true if scheduler has not stepped
    end

    # update to next minibatch (if minibatching)
    next_minibatch = update_minibatch!(ekp)
    return nothing

end


## include the different types of Processes and their exports:

# struct Inversion
export Inversion
include("EnsembleKalmanInversion.jl")

# struct TransformInversion
export TransformInversion
include("EnsembleTransformKalmanInversion.jl")

# struct GaussNewtonInversion
export GaussNewtonInversion
include("GaussNewtonKalmanInversion.jl")

# struct SparseInversion
export SparseInversion
include("SparseEnsembleKalmanInversion.jl")

# struct Sampler
export Sampler
include("EnsembleKalmanSampler.jl")


# struct Unscented
export Unscented
export Gaussian_2d
export construct_initial_ensemble, construct_mean, construct_cov
include("UnscentedKalmanInversion.jl")

# struct Accelerator
include("Accelerators.jl")
