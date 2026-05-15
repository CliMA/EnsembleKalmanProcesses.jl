#Ensemble Kalman Sampler: specific structures and function definitions

export get_sampler_type
export EKS, ALDI

# Sampler type
abstract type SamplerType end

"""
Ensemble Kalman Sampler update variant based on Garbuno-Inigo, Hoffmann, Li, and Stuart (2019).

Select this variant by passing `sampler_type = "eks"` to the `Sampler` constructor.
"""
abstract type EKS <: SamplerType end # Garbuno-Inigo Hoffmann Li Stuart 2019

"""
Affine-invariant Langevin dynamics sampler update variant based on Garbuno-Inigo, Nusken, and Reich (2020).

Select this variant by passing `sampler_type = "aldi"` (the default) to the `Sampler` constructor.
"""
abstract type ALDI <: SamplerType end # Garbuno-Inigo Nusken Reich 2020


"""
An ensemble Kalman Sampler process parameterised by algorithm type `T <: SamplerType` (`ALDI` or `EKS`).

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Sampler{FT <: AbstractFloat, T} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::Vector{FT}
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}}
end

"""
$(TYPEDSIGNATURES)

Construct a `Sampler` process from a `ParameterDistribution` prior.

The `sampler_type` keyword selects the update algorithm: `"aldi"` (default) uses the
affine-invariant Langevin dynamics update (`ALDI`); `"eks"` uses the ensemble Kalman
sampler (`EKS`).

# Arguments
- `prior`: parameter prior distribution used to extract the prior mean and covariance.
- `sampler_type`: one of `"aldi"` (default) or `"eks"`.
"""
function Sampler(prior::ParameterDistribution; sampler_type = "aldi")
    mean_prior = isa(mean(prior), Real) ? [mean(prior)] : Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    st = if sampler_type == "eks"
        EKS
    elseif sampler_type == "aldi"
        ALDI
    else
        throw(ArgumentError("Expected sampler_type in {\"aldi\" (default), \"eks\"}. Received $(sampler_type)."))
    end
    return Sampler{FT, st}(mean_prior, cov_prior)
end


"""
$(TYPEDSIGNATURES)

Return the prior mean vector stored in `process`.
"""
get_prior_mean(process::Sampler) = process.prior_mean

"""
$(TYPEDSIGNATURES)

Return the prior covariance matrix stored in `process`.
"""
get_prior_cov(process::Sampler) = process.prior_cov

"""
$(TYPEDSIGNATURES)

Return the sampler algorithm type (`EKS` or `ALDI`) stored in the `Sampler` process.
"""
get_sampler_type(process::Sampler{T1, T2}) where {T1, T2} = T2

# overload ==
Base.:(==)(s_a::Sampler, s_b::Sampler) =
    get_prior_mean(s_a) == get_prior_mean(s_b) &&
    get_prior_cov(s_a) == get_prior_cov(s_b) &&
    get_sampler_type(s_a) == get_sampler_type(s_b)


function FailureHandler(process::Sampler, method::IgnoreFailures)
    function failsafe_update(ekp, u, g, failed_ens, process)
        u_transposed = permutedims(u, (2, 1))
        g_transposed = permutedims(g, (2, 1))
        u_transposed = eks_update(ekp, u_transposed, g_transposed, process)
        u_new = permutedims(u_transposed, (2, 1))
        return u_new
    end
    return FailureHandler{Sampler, IgnoreFailures}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Return updated parameter vectors by advancing the ensemble one step using the EKS sampler.

Applies a single implicit-explicit stochastic gradient step toward the posterior using
the ensemble Kalman sampler algorithm of Garbuno-Inigo, Hoffmann, Li, and Stuart (2019).
Rows of `u` and `g` correspond to ensemble members (i.e., the transposes of the arrays
stored in `ekp` are expected).

# Arguments
- `ekp`: the `EnsembleKalmanProcess` holding observations, noise covariance, and step size.
- `u`: current ensemble parameter matrix of size `N_ens x N_par`.
- `g`: corresponding forward model output matrix of size `N_ens x N_obs`.
- `process`: `Sampler` process encoding the prior mean, prior covariance, and `EKS` type.

# Method list
$(METHODLIST)
"""
function eks_update(
    ekp::EnsembleKalmanProcess,
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    process::PEKS,
) where {FT <: Real, PEKS <: Sampler{FT, EKS}}
    # TODO: Work with input data as columns

    # u_mean: N_par x 1
    u_mean = mean(u', dims = 2)
    # g_mean: N_obs x 1
    g_mean = mean(g', dims = 2)
    # g_cov: N_obs x N_obs
    g_cov = cov(g, corrected = false)
    add_diagonal_regularization!(g_cov)

    # u_cov: N_par x N_par
    u_cov = cov(u, corrected = false)
    add_diagonal_regularization!(u_cov)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- get_obs(ekp)
    verbose = ekp.verbose
    # D: N_ens x N_ens
    D = (1 / get_N_ens(ekp)) * (E' * safe_linear_solve(get_obs_noise_cov(ekp), R; verbose))

    # Default: dt = 1 / (norm(D) + eps(FT))
    dt = get_Δt(ekp)[end]

    noise = MvNormal(zeros(size(u_cov, 1)), I)
    implicit = safe_linear_solve(
        (I + dt * safe_linear_solve(process.prior_cov', u_cov'; verbose)'),
        (
            u' .- dt * (u' .- u_mean) * D .+
            dt * u_cov * safe_linear_solve(process.prior_cov, process.prior_mean; verbose)
        );
        verbose,
    )

    u = implicit' + sqrt(2 * dt) * (sqrt(u_cov) * rand(get_rng(ekp), noise, get_N_ens(ekp)))'

    return u
end

function eks_update(
    ekp::EnsembleKalmanProcess,
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    process::PALDI,
) where {FT <: Real, PALDI <: Sampler{FT, ALDI}}
    # u_mean: N_par x 1
    u_mean = mean(u', dims = 2)
    # g_mean: N_obs x 1
    g_mean = mean(g', dims = 2)
    # g_cov: N_obs x N_obs
    g_cov = cov(g, corrected = false)
    add_diagonal_regularization!(g_cov)

    # u_cov: N_par x N_par
    u_cov = cov(u, corrected = false)
    add_diagonal_regularization!(u_cov)

    N_ens = get_N_ens(ekp)
    dim_u = length(u_mean)
    # Building tmp matrices for ALDI update:
    U = u' .- u_mean
    E = g' .- g_mean
    R = g' .- get_obs(ekp)
    verbose = ekp.verbose
    # D: N_ens x N_ens
    D = (1 / N_ens) * (E' * safe_linear_solve(get_obs_noise_cov(ekp), R; verbose))
    finite_sample_correction = (dim_u + 1) / N_ens * U
    C_sqrt = 1 / sqrt(N_ens) * U
    # Default: dt = 1 / (norm(D) + eps(FT))
    dt = get_Δt(ekp)[end]

    implicit = safe_linear_solve(
        (I + dt * safe_linear_solve(process.prior_cov', u_cov'; verbose)'),
        (
            u' .- dt * U * D .+ dt * u_cov * safe_linear_solve(process.prior_cov, process.prior_mean; verbose) +
            dt * finite_sample_correction
        );
        verbose,
    )

    u = implicit' + sqrt(2 * dt) * (C_sqrt * randn(get_rng(ekp), (N_ens, N_ens)))' # ensemble-sqrt noise update
    return u
end

function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, P},
    g::AbstractMatrix{FT},
    process::P,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
    kwargs...,
) where {FT, IT, P <: Sampler}

    # u: N_ens x N_par
    # g: N_ens x N_obs
    u_old = get_u_final(ekp)

    fh = get_failure_handler(ekp)

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u_old, g, failed_ens, process)

    return u
end
