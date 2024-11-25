#Ensemble Transform Kalman Inversion: specific structures and function definitions

"""
    TransformInversion <: Process

An ensemble transform Kalman inversion process.

# Fields

$(TYPEDFIELDS)
"""
struct TransformInversion <: Process
    "used to store matrices: buffer[1] = Y' *Γ_inv, buffer[2] = Y' * Γ_inv * Y"
    buffer::AbstractVector
end

TransformInversion() = TransformInversion([])

get_buffer(p::TI) where {TI <: TransformInversion} = p.buffer

function FailureHandler(process::TransformInversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, failed_ens) = etki_update(ekp, u, g)
    return FailureHandler{TransformInversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::TransformInversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the ETKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::TransformInversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] = etki_update(ekp, u[:, successful_ens], g[:, successful_ens])
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
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
    ) where {FT <: Real, IT, CT <: Real}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations.
"""
function etki_update(
    ekp::EnsembleKalmanProcess{FT, IT, TransformInversion},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
) where {FT <: Real, IT}

    y = get_obs(ekp)
    inv_noise_scaling = get_Δt(ekp)[end]

    m = size(u, 2)
    X = FT.((u .- mean(u, dims = 2)) / sqrt(m - 1))
    Y = FT.((g .- mean(g, dims = 2)) / sqrt(m - 1))

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
    Γ_inv = get_obs_noise_cov_inv(ekp, build = false) # returns blocks of Γ_inv 
    γ_sizes = [size(γ_inv, 1) for γ_inv in Γ_inv]
    shift = [0]
    for (γs, γ_inv) in zip(γ_sizes, Γ_inv)
        idx = (shift[1] + 1):(shift[1] + γs)
        tmp[1][1:ys2, idx] = (inv_noise_scaling * γ_inv * Y[idx, :])' # NB: col(Y') * γ_inv = (γ_inv * row(Y))'
        shift[1] = maximum(idx)
    end

    tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

    for i in 1:ys2
        tmp[2][i, i] += 1.0
    end
    Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = inv(I + Y' * Γ_inv * Y)
    w = FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y .- mean(g, dims = 2))) #  w = Ω * Y' * Γ_inv * (y .- g_mean))

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
    ekp::EnsembleKalmanProcess{FT, IT, TransformInversion},
    g::AbstractMatrix{FT},
    process::TransformInversion{FT},
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
    kwargs...,
) where {FT, IT}

    # update only u_idx parameters/ with g_idx data
    # u: length(u_idx) × N_ens   
    # g: lenght(g_idx) × N_ens
    u = get_u_final(ekp)[u_idx, :]
    g = g[g_idx, :]
    obs_noise_cov = ekp.obs_noise_cov[g_idx, g_idx]
    obs_mean = ekp.obs_mean[g_idx]
    # ISSUE. In general this is not true,
    # Gamma_inv = ekp.process.Gamma_inv[g_idx,g_idx]

    N_obs = size(g, 1)
    fh = get_failure_handler(ekp)

    # Scale noise using Δt
    scaled_obs_noise_cov = obs_noise_cov / ekp.Δt[end]

    y = ekp.obs_mean

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, failed_ens)

    return u
end
