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
            u[:, failed_ens] = sample_empirical_gaussian(ekp.rng, u[:, successful_ens], n_failed)
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
    inv_noise_scaling = ekp.Δt[end]

    m = size(u, 2)
    X = FT.((u .- mean(u, dims = 2)) / sqrt(m - 1))
    Y = FT.((g .- mean(g, dims = 2)) / sqrt(m - 1))
    tmp = get_buffer(get_process(ekp)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
    if length(tmp) == 0 #for size 10^4 obs this takes 0.7s
        #normal update to initialize buffer
        Γ_inv = get_obs_noise_cov_inv(ekp) #bottleneck
        Γ_inv *= inv_noise_scaling
        push!(tmp, Y' * Γ_inv) # store in [1]
        push!(tmp, tmp[1] * Y) # store in [2]
        for i in 1:size(Y, 2)
            tmp[2][i, i] += 1.0
        end

        Ω = inv(tmp[2]) # = inv (I + Y' * Γ_inv * Y)

        w = FT.(Ω * tmp[1] * (y .- mean(g, dims = 2)))
        return mean(u, dims = 2) .+ X * (w .+ sqrt(m - 1) * real(sqrt(Ω))) # [N_par × N_ens]
    elseif (size(tmp[1], 1) >= size(Y, 2)) && (size(tmp[1], 2) >= size(Y, 1)) # enough to check tmp[1]
        # for size 10^4 obs this takes 0.005s
        ms_1, ms_2 = size(Y)
        # if buffer is bigger than we need, reuse it and don't build Γ_inv to save some space

        Γ_inv = get_obs_noise_cov_inv(ekp, build = false)
        # these are a hack to find the size of γ_inv when its a UniformScaling and has no size..
        yvec = get_obs(ekp, build = false)
        y_sizes = [length(yi) for yi in yvec]

        shift = [0]
        # block-build: I + Y'*Γ_inv * Y

        # col(Y') * γ_inv = (γ_inv * row(Y))'
        for (ys, γ_inv) in zip(y_sizes, Γ_inv)
            idx = (shift[1] + 1):(shift[1] + ys)
            tmp[1][1:ms_2, idx] = (inv_noise_scaling * γ_inv * Y[idx, :])'
            shift[1] = maximum(idx)
        end
        w = FT.(tmp[1][1:ms_2, 1:ms_1] * (y .- mean(g, dims = 2))) # use tmp = Y' * Γ_inv 

        tmp[2][1:ms_2, 1:ms_2] = tmp[1][1:ms_2, 1:ms_1] * Y

        for i in 1:ms_2
            tmp[2][i, i] += 1.0
        end
        Ω = inv(tmp[2][1:ms_2, 1:ms_2])
        return mean(u, dims = 2) .+ X * (Ω * w .+ sqrt(m - 1) * real(sqrt(Ω))) # [N_par × N_ens]
    else
        #normal update to reinitialize buffer
        Γ_inv = get_obs_noise_cov_inv(ekp) #bottleneck
        Γ_inv *= inv_noise_scaling
        tmp[1] = Y' * Γ_inv
        tmp[2] = tmp[1] * Y
        for i in 1:size(Y, 2)
            tmp[2][i, i] += 1.0
        end
        Ω = inv(tmp[2]) # = inv(I + Y' * Γ_inv * Y)
        w = FT.(Ω * tmp[1] * (y .- mean(g, dims = 2)))
        return mean(u, dims = 2) .+ X * (w .+ sqrt(m - 1) * real(sqrt(Ω))) # [N_par × N_ens]
    end

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
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g` with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, TransformInversion},
    g::AbstractMatrix{FT},
    process::TransformInversion;
    failed_ens = nothing,
    kwargs...
) where {FT, IT}

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)
    N_obs = size(g, 1)
    cov_init = cov(u, dims = 2)

    if ekp.verbose
        if get_N_iterations(ekp) == 0
            @info "Iteration 0 (prior)"
            @info "Covariance trace: $(tr(cov_init))"
        end

        @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(ekp.Δt)))"
    end

    fh = ekp.failure_handler

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, failed_ens)

    # store new parameters (and model outputs)
    push!(ekp.g, DataContainer(g, data_are_columns = true))
    # Store error
    compute_error!(ekp)

    # Diagnostics
    cov_new = cov(u, dims = 2)

    if ekp.verbose
        @info "Covariance-weighted error: $(get_error(ekp)[end])\nCovariance trace: $(tr(cov_new))\nCovariance trace ratio (current/previous): $(tr(cov_new)/tr(cov_init))"
    end

    return u
end
