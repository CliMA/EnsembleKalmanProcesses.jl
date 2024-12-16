#Ensemble Kalman Sampler: specific structures and function definitions

"""
    Sampler{FT<:AbstractFloat,IT<:Int} <: Process

An ensemble Kalman Sampler process.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Sampler{FT <: AbstractFloat} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::Vector{FT}
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}}
end

function Sampler(prior::ParameterDistribution)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    return Sampler{FT}(mean_prior, cov_prior)
end


function FailureHandler(process::Sampler, method::IgnoreFailures)
    function failsafe_update(ekp, u, g, failed_ens)
        u_transposed = permutedims(u, (2, 1))
        g_transposed = permutedims(g, (2, 1))
        u_transposed = eks_update(ekp, u_transposed, g_transposed)
        u_new = permutedims(u_transposed, (2, 1))
        return u_new
    end
    return FailureHandler{Sampler, IgnoreFailures}(failsafe_update)
end

"""
     eks_update(
        ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
    ) where {FT <: Real, IT}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the sampler algorithm.

The current implementation assumes that rows of u and g correspond to
ensemble members, so it requires passing the transpose of the `u` and
`g` arrays associated with ekp.
"""
function eks_update(
    ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
) where {FT <: Real, IT}
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
    process = get_process(ekp)
    implicit =
        (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (process.prior_cov' \ u_cov')') \
        (u' .- Δt * (u' .- u_mean) * D .+ Δt * u_cov * (process.prior_cov \ process.prior_mean))

    u = implicit' + sqrt(2 * Δt) * (sqrt(u_cov) * rand(get_rng(ekp), noise, get_N_ens(ekp)))'

    return u
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}},
        g::AbstractMatrix{FT},
        process::Sampler{FT};
        failed_ens = nothing,
    ) where {FT, IT}

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
    ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}},
    g::AbstractMatrix{FT},
    process::Sampler{FT},
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
) where {FT, IT}

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

    u = fh.failsafe_update(ekp, u_old, g, failed_ens)

    return u
end
