#Ensemble Kalman Inversion: specific structures and function definitions

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

function FailureHandler(process::Inversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens) = eki_update(ekp, u, g, y, obs_noise_cov)
    return FailureHandler{Inversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::Inversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the EKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::Inversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            eki_update(ekp, u[:, successful_ens], g[:, successful_ens], y[:, successful_ens], obs_noise_cov)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(get_rng(ekp), u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{Inversion, SampleSuccGauss}(failsafe_update)
end

"""
     eki_update(
        ekp::EnsembleKalmanProcess{FT, IT, Inversion},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
        y::AbstractMatrix{FT},
        obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
    ) where {FT <: Real, IT, CT <: Real}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the inversion algorithm
from eqns. (4) and (5) of Schillings and Stuart (2017).

Localization is implemented following the `ekp.localizer`.
"""
function eki_update(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    y::AbstractMatrix{FT},
    obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
) where {FT <: Real, IT, CT <: Real}

    cov_est = cov([u; g], dims = 2, corrected = false) # [(N_par + N_obs)×(N_par + N_obs)]

    # Localization
    cov_localized = get_localizer(ekp).localize(cov_est)
    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = try
        FT.((cov_gg + obs_noise_cov) \ (y - g))
    catch e
        if e isa SingularException
            LHS = Matrix{BigFloat}(cov_gg + obs_noise_cov)
            RHS = Matrix{BigFloat}(y - g)
            FT.(LHS \ RHS)
        else
            rethrow(e)
        end
    end
    return u + (cov_ug * tmp) # [N_par × N_ens]  
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, Inversion},
        g::AbstractMatrix{FT},
        process::Inversion;
        deterministic_forward_map::Bool = true,
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to an Inversion process. 

Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - process :: Type of the EKP.
 - deterministic_forward_map :: Whether output `g` comes from a deterministic model.
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g` with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::AbstractMatrix{FT},
    process::Inversion;
    deterministic_forward_map::Bool = true,
    failed_ens = nothing,
) where {FT, IT}

    if !(isa(get_accelerator(ekp), DefaultAccelerator))
        add_stochastic_perturbation = false # doesn't play well with accelerator, but not needed
    else
        add_stochastic_perturbation = deterministic_forward_map
    end
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

        @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(get_Δt(ekp))))"
    end

    fh = get_failure_handler(ekp)

    # Scale noise using Δt
    scaled_obs_noise_cov = get_obs_noise_cov(ekp) / get_Δt(ekp)[end]
    noise = sqrt(scaled_obs_noise_cov) * rand(get_rng(ekp), MvNormal(zeros(N_obs), I), get_N_ens(ekp))

    # Add obs (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic, else just repeat the observation
    y = add_stochastic_perturbation ? (get_obs(ekp) .+ noise) : repeat(get_obs(ekp), 1, get_N_ens(ekp))

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, y, scaled_obs_noise_cov, failed_ens)

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
