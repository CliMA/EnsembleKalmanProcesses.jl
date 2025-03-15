#Ensemble Transform Kalman Inversion: specific structures and function definitions

export get_prior_mean, get_prior_cov, get_impose_prior, get_buffer, get_default_multiplicative_inflation

"""
    TransformInversion <: Process

An ensemble transform Kalman inversion process.

# Fields

$(TYPEDFIELDS)
"""
struct TransformInversion{
    FT <: AbstractFloat,
    NorV <: Union{Nothing, AbstractVector},
    NorAMorUS <: Union{Nothing, AbstractMatrix, UniformScaling},
} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::NorV
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::NorAMorUS
    "flag to explicitly impose the prior mean and covariance during updates"
    impose_prior::Bool
    "if prior is imposed, inflation is often required. This sets a default multiplicative inflation with `s = default_multiplicative_inflation`"
    default_multiplicative_inflation::FT
    "used to store matrices: buffer[1] = Y' *Γ_inv, buffer[2] = Y' * Γ_inv * Y"
    buffer::AbstractVector
end

"""
$(TYPEDSIGNATURES)

Returns the stored `prior_mean` from the TransformInversion process 
"""
get_prior_mean(process::TransformInversion) = process.prior_mean

"""
$(TYPEDSIGNATURES)

Returns the stored `prior_cov` from the TransformInversion process 
"""
get_prior_cov(process::TransformInversion) = process.prior_cov

"""
$(TYPEDSIGNATURES)

Returns the stored `impose_prior` from the TransformInversion process 
"""
get_impose_prior(process::TransformInversion) = process.impose_prior

"""
$(TYPEDSIGNATURES)

Returns the stored `buffer` from the TransformInversion process 
"""
get_buffer(p::TI) where {TI <: TransformInversion} = p.buffer

"""
$(TYPEDSIGNATURES)

Returns the stored `default_multiplicative_inflation` from the TransformInversion process 
"""
get_default_multiplicative_inflation(p::TI) where {TI <: TransformInversion} = p.default_multiplicative_inflation

function TransformInversion(mean_prior, cov_prior; impose_prior = true, default_multiplicative_inflation = 0.0)
    dmi = max(0.0, default_multiplicative_inflation)
    return TransformInversion(mean_prior, cov_prior, impose_prior, dmi, [])
end

"""
$(TYPEDSIGNATURES)

Constructor for prior-enforcing process, (unless `impose_prior` is set false), and `default_multiplicative_inflation` is set to 0.0.
"""
function TransformInversion(prior::ParameterDistribution; impose_prior = true, default_multiplicative_inflation = 0.0)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    return TransformInversion(
        mean_prior,
        cov_prior,
        impose_prior = impose_prior,
        default_multiplicative_inflation = default_multiplicative_inflation,
    )
end

"""
$(TYPEDSIGNATURES)

Constructor for standard non-prior-enforcing `TransformInversion` process
"""
TransformInversion() = TransformInversion(nothing, nothing, false, 0.0, [])

function FailureHandler(process::TransformInversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, u_idx, g_idx, failed_ens) =
        etki_update(ekp, u, g, y, u_idx, g_idx)
    return FailureHandler{TransformInversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::TransformInversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the ETKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::TransformInversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, u_idx, g_idx, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            etki_update(ekp, u[:, successful_ens], g[:, successful_ens], y, u_idx, g_idx)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(get_rng(ekp), u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{TransformInversion, SampleSuccGauss}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations.
"""
function etki_update(
    ekp::EnsembleKalmanProcess{FT, IT, TI},
    u::AM1,
    g::AM2,
    y::AV1,
    u_idx::Vector{Int},
    g_idx::Vector{Int},
) where {
    FT <: Real,
    IT,
    AM1 <: AbstractMatrix,
    AM2 <: AbstractMatrix,
    AV1 <: AbstractVector,
    TI <: TransformInversion,
}
    inv_noise_scaling = get_Δt(ekp)[end]
    m = size(u, 2)

    impose_prior = get_impose_prior(get_process(ekp))
    if impose_prior
        prior_mean = get_prior_mean(get_process(ekp))[u_idx]
        prior_cov_inv = inv(get_prior_cov(get_process(ekp)))[u_idx,u_idx] # take idx later
        # extend y and G
        g_ext = [g; u]
        y_ext = [y; prior_mean]
    else
        y_ext = y
        g_ext = g
    end
    X = FT.((u .- mean(u, dims = 2)) / sqrt(m - 1))
    Y = FT.((g_ext .- mean(g_ext, dims = 2)) / sqrt(m - 1))

    # we have three options with the buffer:
    # (1) in the first iteration, create a buffer
    # (2) if a future iteration requires a smaller buffer, use the existing tmp
    # (3) if a future iteration requires a larger buffer, create this in tmp

    # Create/Enlarge buffers if needed
    tmp = get_buffer(get_process(ekp)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
    ys1, ys2 = size(Y)
    if length(tmp) == 0  # no buffer
        push!(tmp, zeros(ys2, ys1)) # stores Y' * Γ_inv
        push!(tmp, zeros(ys2, ys2)) # stores Y' * Γ_inv * Y
    elseif (size(tmp[1], 1) < ys2) || (size(tmp[1], 2) < ys1) # existing buffer is too small
        tmp[1] = zeros(ys2, ys1)
        tmp[2] = zeros(ys2, ys2)
    end

    ## construct I + Y' * Γ_inv * Y using only blocks γ_inv of Γ_inv
    # left multiply obs_noise_cov_inv in-place (see src/Observations.jl) with the additional index restrictions
    if impose_prior
        lmul_obs_noise_cov_inv!(view(tmp[1]', 1:size(g,1), 1:ys2), ekp, Y[1:size(g,1),:], g_idx) # store in transpose, with view helping reduce allocations
        view(tmp[1]', size(g,1)+1:ys1, 1:ys2) .= prior_cov_inv * Y[size(g,1)+1:end,:]
    else
        lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), ekp, Y, g_idx) # store in transpose, with view helping reduce allocations
    end
    view(tmp[1], 1:ys2, :) .*= inv_noise_scaling

    tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

    for i in 1:ys2
        tmp[2][i, i] += 1.0
    end
    Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = inv(I + Y' * Γ_inv * Y)
    w = FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- mean(g_ext, dims = 2))) #  w = Ω * Y' * Γ_inv * (y .- g_mean))

    return mean(u, dims = 2) .+ X * (w .+ sqrt(m - 1) * real(sqrt(Ω))) # [N_par × N_ens]

end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, TransformInversion},
        g::AbstractMatrix{FT},
        process::TransformInversion;
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to a TransformInversion process. 

Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - process :: Type of the EKP.
 - u_idx :: indices of u to update (see `UpdateGroup`)
 - g_idx :: indices of g,y,Γ with which to update u (see `UpdateGroup`)
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g` with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, TI},
    g::AbstractMatrix{FT},
    process::TransformInversion,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
    kwargs...,
) where {FT, IT, TI <: TransformInversion}

    # update only u_idx parameters/ with g_idx data
    # u: length(u_idx) × N_ens   
    # g: length(g_idx) × N_ens
    u = get_u_final(ekp)[u_idx, :]
    g = g[g_idx, :]
    # get relevant inverse covariance blocks

    N_obs = length(g_idx)

    fh = get_failure_handler(ekp)

    y = get_obs(ekp)[g_idx]

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, y, u_idx, g_idx, failed_ens)

    return u
end
