#Ensemble Transform Kalman Inversion: specific structures and function definitions

export get_prior_mean, get_prior_cov, get_impose_prior, get_buffer

"""
    TransformInversion <: Process

An ensemble transform Kalman inversion process.

# Fields

$(TYPEDFIELDS)
"""
struct TransformInversion{
    NorV <: Union{Nothing, AbstractVector},
    NorAMorUS <: Union{Nothing, AbstractMatrix, UniformScaling},
} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::NorV
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::NorAMorUS
    "flag to explicitly impose the prior mean and covariance during updates"
    impose_prior::Bool
    "used to store matrices: buffer[1] = Y' *Γ_inv, buffer[2] = Y' * Γ_inv * Y"
    buffer::AbstractVector
end

get_prior_mean(process::TransformInversion) = process.prior_mean
get_prior_cov(process::TransformInversion) = process.prior_cov
get_impose_prior(process::TransformInversion) = process.impose_prior
get_buffer(p::TI) where {TI <: TransformInversion} = p.buffer

function TransformInversion(mean_prior, cov_prior; impose_prior = true)
    return TransformInversion(mean_prior, cov_prior, impose_prior, [])
end

"""
$(TYPEDSIGNATURES)

Constructor for prior-enforcing process, (unless keyword is set false) 
"""
function TransformInversion(prior::ParameterDistribution; impose_prior = true)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    return TransformInversion(mean_prior, cov_prior, impose_prior = impose_prior)
end

"""
$(TYPEDSIGNATURES)

Constructor for standard non-prior-enforcing `TransformInversion` process
"""
TransformInversion() = TransformInversion(nothing, nothing, false, [])


function FailureHandler(process::TransformInversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean) =
        etki_update(ekp, u, g, y, obs_noise_cov_inv, onci_idx, prior_mean)
    return FailureHandler{TransformInversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::TransformInversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the ETKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::TransformInversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            etki_update(ekp, u[:, successful_ens], g[:, successful_ens], y, obs_noise_cov_inv, onci_idx, prior_mean)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(get_rng(ekp), u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{TransformInversion, SampleSuccGauss}(failsafe_update)
end

"""
     etki_update(
        ekp::EnsembleKalmanProcess{FT, IT, TransformInversion},
        u::AbstractMatrix,
        g::AbstractMatrix,
        y::AbstractVector,
        obs_noise_cov_inv::AbstractVector,
    ) where {FT <: Real, IT, CT <: Real}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations.
"""
function etki_update(
    ekp::EnsembleKalmanProcess{FT, IT, TI},
    u::AM1,
    g::AM2,
    y::AV1,
    obs_noise_cov_inv::AV2,
    onci_idx::AV3,
    prior_mean::NorAV,
) where {
    FT <: Real,
    IT,
    AM1 <: AbstractMatrix,
    AM2 <: AbstractMatrix,
    AV1 <: AbstractVector,
    AV2 <: AbstractVector,
    AV3 <: AbstractVector,
    TI <: TransformInversion,
    NorAV <: Union{Nothing, AbstractVector},
}
    inv_noise_scaling = get_Δt(ekp)[end]
    m = size(u, 2)

    impose_prior = get_impose_prior(get_process(ekp))
    if impose_prior
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

    # construct I + Y' * Γ_inv * Y using only blocks γ_inv of Γ_inv
    # this loop is very fast for diagonal, slow for nondiagonal
    for (block_idx, local_idx, global_idx) in onci_idx
        γ_inv = obs_noise_cov_inv[block_idx]
        # This is cumbersome, but will retain e.g. diagonal type for matrix manipulations, else indexing converts back to matrix
        if isa(γ_inv, Diagonal) #
            tmp[1][1:ys2, global_idx] = inv_noise_scaling * (γ_inv.diag[local_idx] .* Y[global_idx, :])' # multiple each row of Y by γ_inv element
        else #much slower
            tmp[1][1:ys2, global_idx] = inv_noise_scaling * (γ_inv[local_idx, local_idx] * Y[global_idx, :])' # NB: col(Y') * γ_inv = (γ_inv * row(Y))' row-mult is faster
        end
    end

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
    # g: lenght(g_idx) × N_ens
    u = get_u_final(ekp)[u_idx, :]
    g = g[g_idx, :]
    obs_mean = get_obs(ekp)[g_idx]
    # get relevant inverse covariance blocks
    obs_noise_cov_inv = get_obs_noise_cov_inv(ekp, build = false)# NEVER build=true for this - ruins scaling.
    impose_prior = get_impose_prior(get_process(ekp))
    if impose_prior # these quantitites are truncated to with onci_idx
        prior_mean = get_prior_mean(get_process(ekp))
        prior_cov_inv = inv(get_prior_cov(get_process(ekp)))
    else
        prior_mean = nothing
        prior_cov = nothing
    end


    # need to sweep over local blocks
    if impose_prior
        # extend nois_cov_inv
        push!(obs_noise_cov_inv, prior_cov_inv) # extend noise cov inv to include prior cov inv
        # MUST pop! later
    end
    γ_sizes = [size(γ_inv, 1) for γ_inv in obs_noise_cov_inv]
    prior_flag = repeat([false], length(γ_sizes))
    if impose_prior
        prior_flag[end] = true # needed to swap g_idx to u_idx in loop
    end
    onci_idx = []
    shift = 0
    for (block_id, (γs, pf)) in enumerate(zip(γ_sizes, prior_flag))
        loc_idx = !(pf) ? intersect(1:γs, g_idx .- shift) : intersect(1:γs, u_idx)
        if !(length(loc_idx) == 0)
            push!(onci_idx, (block_id, loc_idx, loc_idx .+ shift))
        end
        shift += γs
    end
    #   obs_noise_cov_inv = [obs_noise_cov_inv[pair[1]][pair[2],pair[2]] for pair in local_intersect] # SLOW
    N_obs = length(g_idx)

    fh = get_failure_handler(ekp)

    y = get_obs(ekp)

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, y, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean)

    if impose_prior
        pop!(obs_noise_cov_inv) # as push! alters the stored observations
    end
    return u
end
