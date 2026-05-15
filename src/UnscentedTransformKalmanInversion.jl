#Unscented Transform Kalman Inversion: specific structures and function definitions

# Many functions are shared with Unscented Kalman Inversion, and are defined in `src/UnscentedKalmanInversion.jl`
"""
A square-root-transform variant of the Unscented Kalman Inversion (UTKI) process.

`TransformUnscented` replaces the classical UKI analysis step with a computationally
efficient square-root/Woodbury formulation, reducing the cost of the covariance update
from O(N_par³) to O(N_ens · N_par²). It shares all constructors with `Unscented`; see
that type for the full constructor reference.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
mutable struct TransformUnscented{FT <: AbstractFloat, IT <: Int} <: Process
    "iterable of vectors of length `N_parameters` containing the parameter mean at each UKI iteration; taken prior to the prediction step and therefore not equal to the sigma-ensemble mean"
    u_mean::Any  # ::Iterable{AbtractVector{FT}}
    "iterable of matrices of size `(N_parameters, N_parameters)` containing the parameter covariance at each UKI iteration; taken prior to the prediction step and therefore not equal to the sigma-ensemble covariance"
    uu_cov::Any  # ::Iterable{AbstractMatrix{FT}}
    "iterable of vectors of length `N_y` containing the predicted observation mean at each UKI iteration"
    obs_pred::Any # ::Iterable{AbstractVector{FT}}
    "sigma-point weights used to shift the mean; vector of length `N_ens` for symmetric sigma points or matrix of size `(N_parameters, N_ens)` for simplex sigma points"
    c_weights::Union{AbstractVector{FT}, AbstractMatrix{FT}}
    "quadrature weights used to reconstruct the mean from the sigma ensemble"
    mean_weights::AbstractVector{FT}
    "quadrature weights used to reconstruct the covariance from the sigma ensemble"
    cov_weights::AbstractVector{FT}
    "number of sigma particles: `2N_parameters + 1` for symmetric or `N_parameters + 2` for simplex"
    N_ens::IT
    "covariance of the artificial evolution noise added during the prediction step"
    Σ_ω::AbstractMatrix{FT}
    "scaling factor for the artificial observation noise covariance"
    Σ_ν_scale::FT
    "regularization parameter controlling shrinkage toward the prior mean (0 < α_reg ≤ 1)"
    α_reg::FT
    "regularization reference vector; defaults to the prior mean"
    r::AbstractVector{FT}
    "frequency at which the evolution covariance `Σ_ω` is updated; 0 disables updates"
    update_freq::IT
    "flag to use augmented-system Tikhonov regularization (Chada et al. 2020, Huang et al. 2022), which imposes the prior during inversion"
    impose_prior::Bool
    "prior mean used for regularization; defaults to the initial mean"
    prior_mean::Any
    "prior covariance used for regularization; defaults to the initial covariance"
    prior_cov::Any
    "current iteration number"
    iter::IT
    "used to store matrices: buffer[1] = Y' * Γ_inv, buffer[2] = Y' * Γ_inv * Y"
    buffer::AbstractVector
end

## Constructors are found in UnscentedKalmanInversion.jl as they use UKI constructors as a base


"""
$(TYPEDSIGNATURES)

Return the computation buffer stored in the `TransformUnscented` process.

The buffer is a two-element vector: `buffer[1]` holds `Y' * Γ⁻¹`
and `buffer[2]` holds `Y' * Γ⁻¹ * Y`, both pre-allocated to avoid
repeated allocation during the analysis step.
"""
get_buffer(p::TU) where {TU <: TransformUnscented} = p.buffer

"""
$(TYPEDSIGNATURES)

Return a `FailureHandler` for `TransformUnscented` that ignores failed ensemble members.

All sigma points are passed to the analysis step regardless of failure status; no
rescaling of weights is performed.
"""
function FailureHandler(process::TransformUnscented, method::IgnoreFailures)
    function failsafe_update(uki, u, g, u_idx, g_idx, failed_ens)
        #perform analysis on the model runs
        update_ensemble_analysis!(uki, u, g, u_idx, g_idx)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(get_process(uki), get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{TransformUnscented, IgnoreFailures}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Provides a failsafe update that
 - computes all means and covariances over the successful sigma points,
 - rescales the mean weights and the off-center covariance weights of the
    successful particles to sum to the same value as the original weight sums.
"""
function FailureHandler(process::TransformUnscented, method::SampleSuccGauss)
    function succ_gauss_analysis!(
        uki::EnsembleKalmanProcess{FT, IT, TU},
        u_p_full,
        g_full,
        u_idx,
        g_idx,
        failed_ens,
    ) where {FT <: Real, IT <: Int, TU <: TransformUnscented}

        process = get_process(uki)
        inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν
        process = get_process(uki)

        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g_full, 2)))

        # update group index of y,g,u
        prior_mean = process.prior_mean[u_idx]
        verbose = uki.verbose
        prior_cov_inv = safe_linear_solve(process.prior_cov[u_idx, u_idx], I(length(u_idx)); verbose) # take idx later
        u_p = u_p_full[u_idx, :]
        y = get_obs(uki)[g_idx]
        g = g_full[g_idx, :]

        u_p_mean = construct_successful_mean(uki, u_p, successful_ens)
        g_mean = construct_successful_mean(uki, g, successful_ens)

        ## extend the state (NB obs_noise_cov_inv already extended)
        if process.impose_prior
            # extend y and G
            g_ext = [g; u_p]
            g_mean_ext = [g_mean; u_p_mean]
            y_ext = [y; prior_mean]
        else
            y_ext = y
            g_mean_ext = g_mean
            g_ext = g
        end
        # (weighted) ensemble square-root: NB assumes u_p_mean/g_mean_ext has be trimmed already
        X = construct_successful_perturbation(uki, u_p, u_p_mean, successful_ens)
        Y = construct_successful_perturbation(uki, g_ext, g_mean_ext, successful_ens)
        # Create/Enlarge buffers if needed
        tmp = get_buffer(get_process(uki)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
        ys1, ys2 = size(Y)
        if length(tmp) == 0  # no buffer
            push!(tmp, zeros(ys2, ys1)) # stores Y' * Γ_inv
            push!(tmp, zeros(ys2, ys2)) # stores Y' * Γ_inv * Y
        elseif (size(tmp[1], 1) < ys2) || (size(tmp[1], 2) < ys1) # existing buffer is too small
            tmp[1] = zeros(ys2, ys1)
            tmp[2] = zeros(ys2, ys2)
        end

        if process.impose_prior
            lmul_obs_noise_cov_inv!(view(tmp[1]', 1:size(g, 1), 1:ys2), uki, Y[1:size(g, 1), :], g_idx) # store in transpose, with view helping reduce allocations
            view(tmp[1]', (size(g, 1) + 1):ys1, 1:ys2) .= process.Σ_ν_scale * prior_cov_inv * Y[(size(g, 1) + 1):end, :]
        else
            lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, g_idx) # store in transpose, with view helping reduce allocations
        end
        view(tmp[1], 1:ys2, :) .*= inv_noise_scaling

        ### Check internal multiplication
        #=
        Σ_ν_inv =  get_obs_noise_cov_inv(uki)[g_idx, g_idx] * get_Δt(uki)[end] / process.Σ_ν_scale
                multmat = Y' * Σ_ν_inv
                @info "diff" norm(multmat - tmp[1][1:ys2,1:ys1])
        =#
        ###

        tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

        for i in 1:ys2
            tmp[2][i, i] += 1.0
        end
        add_diagonal_regularization!(tmp[2][1:ys2, 1:ys2])

        Ω = safe_linear_solve(tmp[2][1:ys2, 1:ys2], I(ys2); verbose) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y      
        u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext)) #  mean update = Ω * Y' * Γ_inv * (y .- g_mean))
        uu_cov = X * Ω * X' # cov update

        ### Check agains true Kalman gain:
        #=
        Gain = X*Y'*(inv(Y*Y'+get_obs_noise_cov(uki) ./ inv_noise_scaling))
        gain_mean = (u_p_mean + Gain * (y_ext .- g_mean_ext))
        @info norm(gain_mean - u_mean)
        =#
        ###

        ########### Save results
        process.obs_pred[end][g_idx] .= g_mean
        process.u_mean[end][u_idx] .= u_mean
        process.uu_cov[end][u_idx, u_idx] .= uu_cov

    end
    function failsafe_update(uki, u, g, u_idx, g_idx, failed_ens)
        #perform analysis on the model runs
        succ_gauss_analysis!(uki, u, g, u_idx, g_idx, failed_ens)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(process, get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{TransformUnscented, SampleSuccGauss}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Perform the UTKI analysis (mean and covariance update) step in-place.

Uses a Woodbury/square-root formulation: constructs the parameter perturbation
matrix `X` and observation perturbation matrix `Y` from the sigma ensemble, then
solves for the updated mean and covariance without forming the full `N_par × N_par`
gain matrix. When `process.impose_prior` is `true` the system is augmented with the
prior constraints before the update.

# Arguments
- `uki`: The `EnsembleKalmanProcess` being updated.
- `u_p_full`: Full parameter ensemble matrix (`N_par × N_ens`).
- `g_full`: Full predicted-observation matrix (`N_obs × N_ens`).
- `u_idx`: Parameter indices belonging to the current `UpdateGroup`.
- `g_idx`: Observation indices belonging to the current `UpdateGroup`.
"""
function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    u_p_full::AM1,
    g_full::AM2,
    u_idx::Vector{Int},
    g_idx::Vector{Int},
) where {FT <: Real, IT <: Int, TU <: TransformUnscented, AM1 <: AbstractMatrix, AM2 <: AbstractMatrix}

    process = get_process(uki)
    # Σ_ν = process.Σ_ν_scale * get_obs_noise_cov(uki) # inefficient
    inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν

    # update group index of y,g,u
    prior_mean = process.prior_mean[u_idx]
    verbose = uki.verbose
    prior_cov_inv = safe_linear_solve(process.prior_cov[u_idx, u_idx], I(length(u_idx)); verbose) # take idx later
    u_p = u_p_full[u_idx, :]
    y = get_obs(uki)[g_idx]
    g = g_full[g_idx, :]

    u_p_mean = construct_mean(uki, u_p)
    g_mean = construct_mean(uki, g)

    if process.impose_prior
        # extend y and G
        g_ext = [g; u_p]
        g_mean_ext = [g_mean; u_p_mean]
        y_ext = [y; prior_mean]
    else
        y_ext = y
        g_mean_ext = g_mean
        g_ext = g
    end

    # sqrt-increments
    X = construct_perturbation(uki, u_p, u_p_mean)
    Y = construct_perturbation(uki, g_ext, g_mean_ext)


    # Create/Enlarge buffers if needed
    tmp = get_buffer(get_process(uki)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
    ys1, ys2 = size(Y)
    if length(tmp) == 0  # no buffer
        push!(tmp, zeros(ys2, ys1)) # stores Y' * Γ_inv
        push!(tmp, zeros(ys2, ys2)) # stores Y' * Γ_inv * Y
    elseif (size(tmp[1], 1) < ys2) || (size(tmp[1], 2) < ys1) # existing buffer is too small
        tmp[1] = zeros(ys2, ys1)
        tmp[2] = zeros(ys2, ys2)
    end

    if process.impose_prior
        lmul_obs_noise_cov_inv!(view(tmp[1]', 1:size(g, 1), 1:ys2), uki, Y[1:size(g, 1), :], g_idx) # store in transpose, with view helping reduce allocations
        view(tmp[1]', (size(g, 1) + 1):ys1, 1:ys2) .= process.Σ_ν_scale * prior_cov_inv * Y[(size(g, 1) + 1):end, :] # 1/Σ_ν_scale is in inv_noise_scaling below, so this will cancel it for this term
    else
        lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, g_idx) # store in transpose, with view helping reduce allocations
    end
    view(tmp[1], 1:ys2, :) .*= inv_noise_scaling

    tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y
    for i in 1:ys2
        tmp[2][i, i] += 1.0
    end
    add_diagonal_regularization!(tmp[2][1:ys2, 1:ys2])
    Ω = safe_linear_solve(tmp[2][1:ys2, 1:ys2], I(ys2); verbose) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y
    u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext))
    uu_cov = X * Ω * X' # cov update 

    ########### Save results
    process.obs_pred[end][g_idx] .= g_mean
    process.u_mean[end][u_idx] .= u_mean
    process.uu_cov[end][u_idx, u_idx] .= uu_cov

end

function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    g::AbstractMatrix{FT},
    process::TU,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    group_idx = 0,
    failed_ens = nothing,
    kwargs...,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    #catch works when g_in non-square 
    u_p_old = get_u_final(uki)
    process = get_process(uki)

    fh = get_failure_handler(uki)

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    # create on first group, then populate later
    if group_idx == 1
        push!(process.obs_pred, zeros(size(g, 1)))
        push!(process.u_mean, zeros(size(u_p_old, 1)))
        push!(process.uu_cov, zeros(size(u_p_old, 1), size(u_p_old, 1)))
    end

    u_p = fh.failsafe_update(uki, u_p_old, g, u_idx, g_idx, failed_ens)

    return u_p
end
