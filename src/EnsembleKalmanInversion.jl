#Ensemble Kalman Inversion: specific structures and function definitions

export get_prior_mean, get_prior_cov, get_impose_prior, get_default_multiplicative_inflation

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion{
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
end

"""
$(TYPEDSIGNATURES)

Returns the stored `prior_mean` from the Inversion process 
"""
get_prior_mean(process::Inversion) = process.prior_mean

"""
$(TYPEDSIGNATURES)

Returns the stored `prior_cov` from the Inversion process 
"""
get_prior_cov(process::Inversion) = process.prior_cov

"""
$(TYPEDSIGNATURES)

Returns the stored `impose_prior` from the Inversion process 
"""
get_impose_prior(process::Inversion) = process.impose_prior

"""
$(TYPEDSIGNATURES)

Returns the stored `default_multiplicative_inflation` from the Inversion process 
"""
get_default_multiplicative_inflation(process::Inversion) = process.default_multiplicative_inflation

function Inversion(mean_prior, cov_prior; impose_prior = true, default_multiplicative_inflation = 1e-3)
    mp = isa(mean_prior, Real) ? [mean_prior] : mean_prior
    dmi = max(0.0, default_multiplicative_inflation)
    return Inversion(mp, cov_prior, impose_prior, dmi)
end

"""
$(TYPEDSIGNATURES)

Constructor for prior-enforcing process, (unless `impose_prior` is set false), and `default_multiplicative_inflation` is set to 1e-3. 
"""
function Inversion(prior::ParameterDistribution; impose_prior = true, default_multiplicative_inflation = 1e-3)
    mean_prior = isa(mean(prior), Real) ? [mean(prior)] : Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    return Inversion(
        mean_prior,
        cov_prior,
        impose_prior = impose_prior,
        default_multiplicative_inflation = default_multiplicative_inflation,
    )
end

"""
$(TYPEDSIGNATURES)

Constructor for standard non-prior-enforcing `Inversion` process
"""
Inversion() = Inversion(nothing, nothing, false, 0.0)

function FailureHandler(process::Inversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens, prior_mean, scaled_prior_cov) =
        eki_update(ekp, u, g, y, obs_noise_cov, prior_mean, scaled_prior_cov)
    return FailureHandler{Inversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::Inversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the EKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::Inversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens, prior_mean, scaled_prior_cov)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] = eki_update(
            ekp,
            u[:, successful_ens],
            g[:, successful_ens],
            y[:, successful_ens],
            obs_noise_cov,
            prior_mean,
            scaled_prior_cov,
        )
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
    ekp::EnsembleKalmanProcess{FT, IT, II},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    y::AbstractMatrix{FT},
    obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
    prior_mean::NorAV,
    prior_cov::NorAM,
) where {
    FT <: Real,
    IT,
    CT <: Real,
    II <: Inversion,
    NorAV <: Union{Nothing, AbstractVector},
    NorAM <: Union{Nothing, AbstractMatrix},
}

    impose_prior = get_impose_prior(get_process(ekp))
    if impose_prior
        g_ext = [g; u]
        y_ext = [y; repeat(prior_mean, 1, size(y, 2))]

        cov_est = cov([u; g_ext], dims = 2, corrected = false) # [(N_par + N_obs)×(N_par + N_obs)]

        add_diagonal_regularization!(cov_est)
        # Localization - a function taking in (cov, float-type, n_par, n_obs, n_ens)
        cov_localized = get_localizer(ekp).localize(cov_est, FT, size(u, 1), size(g_ext, 1), size(u, 2))
        cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))

        dim1 = size(obs_noise_cov, 1)
        dim2 = size(prior_cov, 1)
        obs_noise_cov_ext = zeros(dim1 + dim2, dim1 + dim2)
        obs_noise_cov_ext[1:dim1, 1:dim1] += obs_noise_cov
        obs_noise_cov_ext[(dim1 + 1):(dim1 + dim2), (dim1 + 1):(dim1 + dim2)] += prior_cov

    else # no extension
        g_ext = g
        y_ext = y
        cov_est = cov([u; g], dims = 2, corrected = false) # [(N_par + N_obs)×(N_par + N_obs)]
        add_diagonal_regularization!(cov_est)

        # Localization - a function taking in (cov, float-type, n_par, n_obs, n_ens)
        cov_localized = get_localizer(ekp).localize(cov_est, FT, size(u, 1), size(g, 1), size(u, 2))
        cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))

        obs_noise_cov_ext = obs_noise_cov
    end



    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = FT.(safe_linear_solve(cov_gg + obs_noise_cov_ext, y_ext - g_ext))
    return u + (cov_ug * tmp) # [N_par × N_ens]  
end

"""
$(TYPEDSIGNATURES)

Updates the ensemble according to an Inversion process. 

Inputs:
 - `ekp` :: The EnsembleKalmanProcess to update.
 - `g` :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - `process` :: Type of the EKP.
 - `u_idx` :: indices of u to update (see `UpdateGroup`)
 - `g_idx` :: indices of g,y,Γ with which to update u (see `UpdateGroup`)
 - `deterministic_forward_map` :: Whether output `g` comes from a deterministic model.
 - `failed_ens` :: Indices of failed particles. If nothing, failures are computed as columns of `g` with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, II},
    g::AbstractMatrix{FT},
    process::Inversion,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    deterministic_forward_map::Bool = true,
    failed_ens = nothing,
    kwargs...,
) where {FT, IT, II <: Inversion}

    if !(isa(get_accelerator(ekp), DefaultAccelerator))
        add_stochastic_perturbation = false # doesn't play well with accelerator, but not needed
    else
        add_stochastic_perturbation = deterministic_forward_map
    end
    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)[u_idx, :]
    g = g[g_idx, :]
    N_obs = length(g_idx)
    obs_noise_cov = get_obs_noise_cov(ekp)[g_idx, g_idx]
    obs_mean = get_obs(ekp)[g_idx]
    impose_prior = get_impose_prior(get_process(ekp))
    if impose_prior
        prior_mean = get_prior_mean(get_process(ekp))[u_idx]
        scaled_prior_cov = get_prior_cov(get_process(ekp))[u_idx, u_idx] / get_Δt(ekp)[end]
    else
        prior_mean = nothing
        scaled_prior_cov = nothing
    end

    fh = get_failure_handler(ekp)

    # Scale noise using Δt
    scaled_obs_noise_cov = obs_noise_cov / get_Δt(ekp)[end]
    noise = sqrt(scaled_obs_noise_cov) * rand(get_rng(ekp), MvNormal(zeros(N_obs), I), get_N_ens(ekp))

    # Add obs (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic, else just repeat the observation
    y = add_stochastic_perturbation ? (obs_mean .+ noise) : repeat(obs_mean, 1, get_N_ens(ekp))

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, y, scaled_obs_noise_cov, failed_ens, prior_mean, scaled_prior_cov)

    return u
end
