#Ensemble Kalman Sampler: specific structures and function definitions

export get_sampler_type
export EKS, ALDI

# Sampler type 
abstract type SamplerType end
abstract type EKS <: SamplerType end # Garbuno-Iñigo Hoffmann Li Stuart 2019
abstract type ALDI <: SamplerType end # Garbuno-Iñigo Nüsken Reich 2020


"""
    Sampler{FT<:AbstractFloat, T <:SamplerType} <: Process

An ensemble Kalman Sampler process. with type Sampler Type (e.g., ALDI or EKS).

# Constructor
Sampler(prior::ParameterDistribution) # ALDI update (sampler_type="aldi")
Sampler(prior::ParameterDistribution; sampler_type = "eks") # EKS update

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

function Sampler(prior::ParameterDistribution; sampler_type = "aldi")
    mean_prior = isa(mean(prior), Real) ? [mean(prior)] : Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    st = if sampler_type == "eks"
        EKS
    elseif sampler_type == "aldi"
        ALDI
    else
        throw(ArgumentError("Expected sampler_type ∈ {\"aldi\" (default), \"eks\"}. Received $(sampler_type)."))
    end
    return Sampler{FT, st}(mean_prior, cov_prior)
end


get_prior_mean(process::Sampler) = process.prior_mean
get_prior_cov(process::Sampler) = process.prior_cov
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

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the sampler algorithm
of (Garbuno-Iñigo Hoffmann Li Stuart 2019)

The current implementation assumes that rows of u and g correspond to
ensemble members, so it requires passing the transpose of the `u` and
`g` arrays associated with ekp.
"""
function eks_update(
    ekp::EnsembleKalmanProcess,
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    process::PEKS,
) where {FT <: Real, PEKS <: Sampler{FT, EKS}}
    # TODO: Work with input data as columns

    # u_mean: N_par × 1
    u_mean = mean(u', dims = 2)
    # g_mean: N_obs × 1
    g_mean = mean(g', dims = 2)
    # g_cov: N_obs × N_obs
    g_cov = cov(g, corrected = false)
    # u_cov: N_par × N_par
    u_cov = cov(u, corrected = false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- get_obs(ekp)
    # D: N_ens × N_ens
    D = (1 / get_N_ens(ekp)) * (E' * (get_obs_noise_cov(ekp) \ R))

    # Default: Δt = 1 / (norm(D) + eps(FT))
    Δt = get_Δt(ekp)[end]

    noise = MvNormal(zeros(size(u_cov, 1)), I)
    implicit =
        (I + Δt * (process.prior_cov' \ u_cov')') \
        (u' .- Δt * (u' .- u_mean) * D .+ Δt * u_cov * (process.prior_cov \ process.prior_mean))

    u = implicit' + sqrt(2 * Δt) * (sqrt(u_cov) * rand(get_rng(ekp), noise, get_N_ens(ekp)))'

    return u
end

"""
$(TYPEDSIGNATURES)

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the sampler algorithm
of (Garbuno-Iñigo Nüsken Reich 2020)

The current implementation assumes that rows of u and g correspond to
ensemble members, so it requires passing the transpose of the `u` and
`g` arrays associated with ekp.
"""
function eks_update(
    ekp::EnsembleKalmanProcess,
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    process::PALDI,
) where {FT <: Real, PALDI <: Sampler{FT, ALDI}}
    # u_mean: N_par × 1
    u_mean = mean(u', dims = 2)
    # g_mean: N_obs × 1
    g_mean = mean(g', dims = 2)
    # g_cov: N_obs × N_obs
    g_cov = cov(g, corrected = false)
    # u_cov: N_par × N_par
    u_cov = cov(u, corrected = false)

    N_ens = get_N_ens(ekp)
    dim_u = length(u_mean)
    # Building tmp matrices for ALDI update:
    U = u' .- u_mean
    E = g' .- g_mean
    R = g' .- get_obs(ekp)
    # D: N_ens × N_ens
    D = (1 / N_ens) * (E' * (get_obs_noise_cov(ekp) \ R))
    finite_sample_correction = (dim_u + 1) / N_ens * U
    C_sqrt = 1 / sqrt(N_ens) * U
    # Default: Δt = 1 / (norm(D) + eps(FT))
    Δt = get_Δt(ekp)[end]

    implicit =
        (I + Δt * (process.prior_cov' \ u_cov')') \
        (u' .- Δt * U * D .+ Δt * u_cov * (process.prior_cov \ process.prior_mean) + Δt * finite_sample_correction)

    u = implicit' + sqrt(2 * Δt) * (C_sqrt * randn(get_rng(ekp), (N_ens, N_ens)))' # ensemble-sqrt noise update
    return u
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT,ST}},
        g::AbstractMatrix{FT},
        process::Sampler{FT, ST};
        failed_ens = nothing,
    ) where {FT, IT, ST}

Updates the ensemble according to a Sampler process. 

Inputs:
 - `ekp` :: The EnsembleKalmanProcess to update.
 - `g` :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - `process` :: Type of the EKP.
 - `u_idx` :: indices of u to update (see `UpdateGroup`)
 - `g_idx` :: indices of g,y,Γ with which to update u (see `UpdateGroup`)
 - `failed_ens` :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, P},
    g::AbstractMatrix{FT},
    process::P,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
    kwargs...,
) where {FT, IT, P <: Sampler}

    # u: N_ens × N_par
    # g: N_ens × N_obs
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
