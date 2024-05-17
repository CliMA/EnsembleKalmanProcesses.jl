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
function FailureHandler(::Inversion, ::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens)
        (failed_ens, sample_transform, sample_dim) = get_correlations(ekp.level_scheduler, failed_ens)
        u = eki_update(ekp, u, g, y, obs_noise_cov; ignored_indices = failed_ens)
        if !isempty(failed_ens)
            new_samples = sample_empirical_gaussian(ekp.rng, ekp, u, sample_dim; ignored_indices = failed_ens)
            u[:, failed_ens] = new_samples[:, sample_transform]
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
        obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}};
        ignored_indices = [],
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
    obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}};
    ignored_indices = [],
) where {FT <: Real, IT, CT <: Real}

    cov_est = compute_cov(ekp, [u; g]; corrected = false, ignored_indices) # [(N_par + N_obs)×(N_par + N_obs)]

    # Localization
    cov_localized = ekp.localizer.localize(cov_est)

    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))
    cov_gg = posdef(cov_gg)

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

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)
    N_obs = size(g, 1)

    if ekp.verbose
        cov_init = compute_cov(ekp, u; corrected = true)

        if get_N_iterations(ekp) == 0
            @info "Iteration 0 (prior)"
            @info "Covariance trace: $(tr(cov_init))"
        end

        @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(ekp.Δt)))"
    end

    fh = ekp.failure_handler

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    independent_noise_dim = get_N_indep(ekp.level_scheduler)
    noise = scaled_obs_noise_cov * rand(ekp.rng, MvNormal(zeros(N_obs), I), independent_noise_dim)
    noise = transform_noise(ekp.level_scheduler, noise)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

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

    if ekp.verbose
        # Diagnostics
        cov_new = compute_cov(ekp, u; corrected = true)

        @info "Covariance-weighted error: $(get_error(ekp)[end])\nCovariance trace: $(tr(cov_new))\nCovariance trace ratio (current/previous): $(tr(cov_new)/tr(cov_init))"
    end

    return u
end
