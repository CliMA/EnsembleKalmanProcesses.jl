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
export compute_error!
export update_ensemble!
export sample_empirical_gaussian, split_indices_by_success
export SampleSuccGauss, IgnoreFailures, FailureHandler

abstract type Process end
#specific Processes and their exports are included after the general definitions

# Failure handlers
abstract type FailureHandlingMethod end

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

## begin general constructor and function definitions

"""
    EnsembleKalmanProcess{FT <: AbstractFloat, IT <: Int, P <: Process}

Structure that is used in Ensemble Kalman processes.

# Fields

$(TYPEDFIELDS)

# Generic constructor

    EnsembleKalmanProcess(
        params::AbstractMatrix{FT},
        obs_mean,
        obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
        process::P;
        Δt = FT(1),
        rng::AbstractRNG = Random.GLOBAL_RNG,
        failure_handler_method::FM = IgnoreFailures(),
        localization_method::LM = NoLocalization(),
        verbose::Bool = false,
    ) where {FT <: AbstractFloat, P <: Process, FM <: FailureHandlingMethod, LM <: LocalizationMethod}

Inputs:

 - `params`                 :: Initial parameter ensemble
 - `obs_mean`               :: Vector of observations
 - `obs_noise_cov`          :: Noise covariance associated with the observations `obs_mean`
 - `process`                :: Algorithm used to evolve the ensemble
 - `Δt`                     :: Initial time step or learning rate
 - `rng`                    :: Random number generator
 - `failure_handler_method` :: Method used to handle particle failures
 - `localization_method`    :: Method used to localize sample covariances
 - `verbose`                :: Whether to print diagnostic information

# Other constructors:

$(METHODLIST)
"""
struct EnsembleKalmanProcess{FT <: AbstractFloat, IT <: Int, P <: Process}
    "array of stores for parameters (`u`), each of size [`N_par × N_ens`]"
    u::Array{DataContainer{FT}}
    "vector of the observed vector size [`N_obs`]"
    obs_mean::Vector{FT}
    "covariance matrix of the observational noise, of size [`N_obs × N_obs`]"
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}}
    "ensemble size"
    N_ens::IT
    "Array of stores for forward model outputs, each of size  [`N_obs × N_ens`]"
    g::Array{DataContainer{FT}}
    "vector of errors"
    err::Vector{FT}
    "vector of timesteps used in each EK iteration"
    Δt::Vector{FT}
    "the particular EK process (`Inversion` or `Sampler` or `Unscented` or `SparseInversion`)"
    process::P
    "Random number generator object (algorithm + seed) used for sampling and noise, for reproducibility. Defaults to `Random.GLOBAL_RNG`."
    rng::AbstractRNG
    "struct storing failsafe update directives, implemented for (`Inversion`, `SparseInversion`, `Unscented`)"
    failure_handler::FailureHandler
    "Localization kernel, implemented for (`Inversion`, `SparseInversion`, `Unscented`)"
    localizer::Localizer
    "Whether to print diagnostics for each EK iteration"
    verbose::Bool
end

function EnsembleKalmanProcess(
    params::AbstractMatrix{FT},
    obs_mean,
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    process::P;
    Δt = FT(1),
    rng::AbstractRNG = Random.GLOBAL_RNG,
    failure_handler_method::FM = IgnoreFailures(),
    localization_method::LM = NoLocalization(),
    verbose::Bool = false,
) where {FT <: AbstractFloat, P <: Process, FM <: FailureHandlingMethod, LM <: LocalizationMethod}

    #initial parameters stored as columns
    init_params = DataContainer(params, data_are_columns = true)

    # dimensionality
    N_par, N_ens = size(init_params) #stored with data as columns
    N_obs = length(obs_mean)

    IT = typeof(N_ens)
    #store for model evaluations
    g = []
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])
    # failure handler
    fh = FailureHandler(process, failure_handler_method)
    # localizer
    loc = Localizer(localization_method, N_par, N_obs, N_ens, FT)

    if verbose
        @info "Initializing ensemble Kalman process of type $(nameof(typeof(process)))\nNumber of ensemble members: $(N_ens)\nLocalization: $(nameof(typeof(localization_method)))\nFailure handler: $(nameof(typeof(failure_handler_method)))"
    end

    EnsembleKalmanProcess{FT, IT, P}(
        [init_params],
        obs_mean,
        obs_noise_cov,
        N_ens,
        g,
        err,
        Δt,
        process,
        rng,
        fh,
        loc,
        verbose,
    )
end



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
    get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess, iteration::IT)

Returns the constrained parameters at the given iteration.
"""
function get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess, iteration::IT) where {IT <: Integer}
    return transform_unconstrained_to_constrained(prior, get_u(ekp, iteration))
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
    get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess)

Returns the unconstrained parameters from all iterations. The outer dimension is given by the number of iterations.
"""
get_ϕ(prior::ParameterDistribution, ekp::EnsembleKalmanProcess) =
    transform_unconstrained_to_constrained(prior, get_u(ekp))

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
    get_u_final(ekp::EnsembleKalmanProcess, return_array=true)

Get the unconstrained parameters at the last iteration, returning a `DataContainer` Object if `return_array` is false.
"""
function get_u_final(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_u(ekp, size(ekp.u, 1)) : ekp.u[end]
end

"""
    get_u_prior(ekp::EnsembleKalmanProcess, return_array=true)

Get the unconstrained parameters as drawn from the prior, returning a `DataContainer` Object if `return_array` is false.
"""
function get_u_prior(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_u(ekp, 1) : ekp.u[1]
end

"""
    get_g_final(ekp::EnsembleKalmanProcess, return_array=true)

Get forward model outputs at the last iteration, returns a `DataContainer` Object if `return_array` is false.
"""
function get_g_final(ekp::EnsembleKalmanProcess; return_array = true)
    return return_array ? get_g(ekp, size(ekp.g, 1)) : ekp.g[end]
end

"""
    get_ϕ_final(ekp::EnsembleKalmanProcess)

Get the constrained parameters at the last iteration.
"""
get_ϕ_final(prior::ParameterDistribution, ekp::EnsembleKalmanProcess) =
    transform_unconstrained_to_constrained(prior, get_u_final(ekp))

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

"""
    get_process(ekp::EnsembleKalmanProcess)
Return process type of EnsembleKalmanProcess.
"""
function get_process(ekp::EnsembleKalmanProcess)
    return ekp.process
end

"""
    get_localizer(ekp::EnsembleKalmanProcess)
Return localizer type of EnsembleKalmanProcess.
"""
function get_localizer(ekp::EnsembleKalmanProcess)
    return Localizers.get_localizer(ekp.localizer)
end


"""
    construct_initial_ensemble(
        rng::AbstractRNG,
        prior::ParameterDistribution,
        N_ens::IT;
        rng_seed::Union{IT, Nothing} = nothing,
    ) where {IT <: Int}
    construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; kwargs...) where {IT <: Int}

Construct the initial parameters, by sampling `N_ens` samples from specified
prior distribution. Returned with parameters as columns.
"""
function construct_initial_ensemble(
    rng::AbstractRNG,
    prior::ParameterDistribution,
    N_ens::IT;
    rng_seed::Union{IT, Nothing} = nothing,
) where {IT <: Int}
    # Ensuring reproducibility of the sampled parameter values: 
    # re-seed the rng *only* if we're given a seed
    if rng_seed !== nothing
        rng = Random.seed!(rng, rng_seed)
    end
    return sample(rng, prior, N_ens) #of size [dim(param space) N_ens]
end
# first arg optional; defaults to GLOBAL_RNG (as in Random, StatsBase)
construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; kwargs...) where {IT <: Int} =
    construct_initial_ensemble(Random.GLOBAL_RNG, prior, N_ens; kwargs...)

"""
    compute_error!(ekp::EnsembleKalmanProcess)

Computes the covariance-weighted error of the mean forward model output, `(ḡ - y)'Γ_inv(ḡ - y)`.
The error is stored within the `EnsembleKalmanProcess`.
"""
function compute_error!(ekp::EnsembleKalmanProcess)
    mean_g = dropdims(mean(get_g_final(ekp), dims = 2), dims = 2)
    diff = ekp.obs_mean - mean_g
    X = ekp.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end

"""
    get_error(ekp::EnsembleKalmanProcess)

Returns the mean forward model output error as a function of algorithmic time.
"""
get_error(ekp::EnsembleKalmanProcess) = ekp.err

function set_Δt!(ekp::EnsembleKalmanProcess, Δt_new::T) where {T}
    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end
end

"""
    sample_empirical_gaussian(
        u::AbstractMatrix{FT},
        n::IT;
        inflation::Union{FT, Nothing} = nothing,
    ) where {FT <: Real, IT <: Int}

Returns `n` samples from an empirical Gaussian based on point estimates `u`, adding inflation
if the covariance is singular.
"""
function sample_empirical_gaussian(
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
    return rand(MvNormal(mean_u_new[:], cov_u_new), n)
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
    multiplicative_inflation!(
        ekp::EnsembleKalmanProcess;
        s,
    ) where {FT, IT}
Applies multiplicative noise to particles.
Inputs:
    - ekp :: The EnsembleKalmanProcess to update.
    - s :: Scaling factor for time step in multiplicative perturbation.
"""
function multiplicative_inflation!(ekp::EnsembleKalmanProcess; s::FT = 1.0) where {FT <: Real}

    scaled_Δt = s * ekp.Δt[end]

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
    additive_inflation!(
        ekp::EnsembleKalmanProcess;
        use_prior_cov::Bool = false,
        s::FT = 1.0,
    ) where {FT <: Real}
Applies additive Gaussian noise to particles. Noise is drawn from normal distribution with 0 mean
and scaled parameter covariance. If use_prior_cov=false (default), scales parameter covariance matrix from
current ekp iteration. Otherwise, scales parameter covariance of initial ensemble.
Inputs:
    - ekp :: The EnsembleKalmanProcess to update.
    - s :: Scaling factor for time step in additive perturbation.
    - use_prior_cov :: Bool specifying whether to use prior covariance estimate for additive inflation.
        If false (default), parameter covariance from the current iteration is used.
"""
function additive_inflation!(ekp::EnsembleKalmanProcess; use_prior_cov::Bool = false, s::FT = 1.0) where {FT <: Real}

    scaled_Δt = s * ekp.Δt[end]

    if scaled_Δt >= 1.0
        error(string("Scaled time step: ", scaled_Δt, " is >= 1.0", "\nChange s or EK time step."))
    end

    Σ = use_prior_cov ? get_u_cov_prior(ekp) : get_u_cov_final(ekp)

    u = get_u_final(ekp)
    # add multivariate noise with 0 mean and scaled covariance
    noise_multivariate = MvNormal((scaled_Δt / (1 - scaled_Δt)) .* Σ)
    u_updated = u + rand(noise_multivariate, size(u, 2))
    ekp.u[end] = DataContainer(u_updated, data_are_columns = true)
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess,
        g::AbstractMatrix{FT};
        multiplicative_inflation::Bool = false,
        additive_inflation::Bool = false,
        use_prior_cov::Bool = false,
        s::FT = 0.0,
        ekp_kwargs...,
    ) where {FT, IT}
Updates the ensemble according to an Inversion process.
Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - multiplicative_inflation :: Flag indicating whether to use multiplicative inflation.
 - additive_inflation :: Flag indicating whether to use additive inflation.
 - use_prior_cov :: Bool specifying whether to use prior covariance estimate for additive inflation.
        If false (default), parameter covariance from the current iteration is used.
 - s :: Scaling factor for time step in inflation step.
 - ekp_kwargs :: Keyword arguments to pass to standard ekp update_ensemble!.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess,
    g::AbstractMatrix{FT};
    multiplicative_inflation::Bool = false,
    additive_inflation::Bool = false,
    use_prior_cov::Bool = false,
    s::FT = 0.0,
    ekp_kwargs...,
) where {FT, IT}

    update_ensemble!(ekp, g, get_process(ekp); ekp_kwargs...)
    if s > 0.0
        multiplicative_inflation ? multiplicative_inflation!(ekp; s = s) : nothing
        additive_inflation ? additive_inflation!(ekp; use_prior_cov = use_prior_cov, s = s) : nothing
    end
end

## include the different types of Processes and their exports:

# struct Inversion
export Inversion
export find_ekp_stepsize
include("EnsembleKalmanInversion.jl")

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
