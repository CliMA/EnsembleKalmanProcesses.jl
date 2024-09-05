#Gauss Newton Kalman Inversion: specific structures and function definitions

"""
    GaussNewtonInversion <: Process

A Gauss Newton Kalman Inversion process
"""
struct GaussNewtonInversion{VV <: AbstractVector, AMorUS <: Union{AbstractMatrix, UniformScaling}} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::VV
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::AMorUS

end

# constructors
function GaussNewtonInversion(prior::ParameterDistribution)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    return GaussNewtonInversion(mean_prior, cov_prior)
end

# failure handling
function FailureHandler(process::GaussNewtonInversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens) = gnki_update(ekp, u, g, y, obs_noise_cov)
    return FailureHandler{GaussNewtonInversion, IgnoreFailures}(failsafe_update)
end


"""
    FailureHandler(process::GaussNewtonInversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the GNKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::GaussNewtonInversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            gnki_update(ekp, u[:, successful_ens], g[:, successful_ens], y[:, successful_ens], obs_noise_cov)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(get_rng(ekp), u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{GaussNewtonInversion, SampleSuccGauss}(failsafe_update)
end

"""
     gnki_update(
        ekp::EnsembleKalmanProcess{FT, IT, GaussNewtonInversion},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
        y::AbstractMatrix{FT},
        scaled_obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
    ) where {FT <: Real, IT, CT <: Real}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the .

Localization is implemented following the `ekp.localizer`.
"""
function gnki_update(
    ekp::EnsembleKalmanProcess{FT, IT, GNI},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    y::AbstractMatrix{FT},
    scaled_obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
) where {FT <: Real, IT, CT <: Real, GNI <: GaussNewtonInversion}
    N_ens_successful = size(g, 2)
    cov_est = cov([u; g], dims = 2, corrected = false) # [(N_par + N_obs)×(N_par + N_obs)]

    cov_localized = get_localizer(ekp).localize(cov_est)
    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))
    process = get_process(ekp)
    prior_mean = process.prior_mean
    prior_cov = process.prior_cov

    #perturbed mean
    Δt = get_Δt(ekp)[end]
    N_par = size(u, 1)
    scaled_prior_cov = 2 * prior_cov / Δt

    m_noise = sqrt(scaled_prior_cov) * rand(get_rng(ekp), MvNormal(zeros(N_par), I), N_ens_successful)
    m = (prior_mean .+ m_noise)
    obs_noise_cov = scaled_obs_noise_cov * Δt / 2

    prior_contribution = -cov_ug' * (cov_uu \ (m .- u))

    data_contribution = y .- g
    A = data_contribution + prior_contribution
    # solve P (Cᵘᵍ)ᵀ (Cᵘᵘ)⁻¹ ( (Cᵘᵍ)ᵀ(Cᵘᵘ)⁻¹ P (Cᵘᵘ)⁻¹Cᵘᵍ + Γ)⁻¹ * A

    # Q =       
    Q = cov_ug' * (cov_uu \ (prior_cov * (cov_uu \ cov_ug)))

    update = prior_cov * (cov_uu \ (cov_ug * ((Q + obs_noise_cov) \ A)))

    return (1 - Δt) * u + Δt * (m .+ update)

end




"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, GaussNewtonInversion},
        g::AbstractMatrix{FT},
        process::GaussNewtonInversion;
        deterministic_forward_map::Bool = true,
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to an GaussNewtonInversion process. 

Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - process :: Type of the EKP.
 - deterministic_forward_map :: Whether output `g` comes from a deterministic model.
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g` with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, GNI},
    g::AbstractMatrix{FT},
    process::GNI;
    deterministic_forward_map::Bool = true,
    failed_ens = nothing,
) where {FT, IT, GNI <: GaussNewtonInversion}

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

    # Scale noise using Δt / 2
    scaled_obs_noise_cov = 2 * get_obs_noise_cov(ekp) / get_Δt(ekp)[end]
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
