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
    R = g' .- ekp.obs_mean
    # D: N_ens × N_ens
    D = (1 / ekp.N_ens) * (E' * (ekp.obs_noise_cov \ R))

    # Default: Δt = 1 / (norm(D) + eps(FT))
    Δt = ekp.Δt[end]

    noise = MvNormal(u_cov)

    implicit =
        (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov' \ u_cov')') \
        (u' .- Δt * (u' .- u_mean) * D .+ Δt * u_cov * (ekp.process.prior_cov \ ekp.process.prior_mean))

    u = implicit' + sqrt(2 * Δt) * rand(ekp.rng, noise, ekp.N_ens)'

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
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - process :: Type of the EKP.
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}},
    g::AbstractMatrix{FT},
    process::Sampler{FT};
    failed_ens = nothing,
) where {FT, IT}

    # u: N_ens × N_par
    # g: N_ens × N_obs
    u_old = get_u_final(ekp)
    cov_init = get_u_cov_final(ekp)

    fh = ekp.failure_handler

    if ekp.verbose
        if get_N_iterations(ekp) == 0
            @info "Iteration 0 (prior)"
            @info "Covariance trace: $(tr(cov_init))"
        end

        @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(ekp.Δt)))"
    end

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u_old, g, failed_ens)

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns = true))
    push!(ekp.g, DataContainer(g, data_are_columns = true))
    # u_old is N_ens × N_par, g is N_ens × N_obs,
    # but stored in data container with N_ens as the 2nd dim

    compute_error!(ekp)

    # Diagnostics
    cov_new = get_u_cov_final(ekp)

    if ekp.verbose
        @info "Covariance-weighted error: $(get_error(ekp)[end])\nCovariance trace: $(tr(cov_new))\nCovariance trace ratio (current/previous): $(tr(cov_new)/tr(cov_init))"
    end
end
