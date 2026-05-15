#Unscented Kalman Inversion: specific structures and function definitions

"""
An Unscented Kalman Inversion process.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
mutable struct Unscented{FT <: AbstractFloat, IT <: Int} <: Process
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
end

"""
$(TYPEDSIGNATURES)

Construct an `Unscented` process from an initial mean and covariance.

# Arguments
- `u0_mean`: initial parameter mean vector.
- `uu0_cov`: initial parameter covariance matrix.
- `α_reg`: regularization parameter controlling shrinkage toward the prior mean
  (0 < `α_reg` ≤ 1); default `1.0` disables regularization.
- `update_freq`: frequency at which the evolution covariance is updated; set to `0`
  for non-identifiable (ill-posed) problems where the covariance tracks parameter
  sensitivity rather than posterior uncertainty, or to `1` (or any positive integer)
  for identifiable problems where the covariance converges to the posterior covariance.
- `modified_unscented_transform`: if `true`, applies the modified UKI quadrature
  from Huang et al. (2021).
- `impose_prior`: if `true`, uses the augmented-system Tikhonov regularization
  (Chada et al. 2020, Huang et al. 2022), which imposes the prior and is recommended
  for ill-posed problems; disables other regularization automatically.
- `prior_mean`: prior mean for regularization; defaults to `u0_mean` when `impose_prior=true`.
- `prior_cov`: prior covariance for regularization; defaults to `uu0_cov` when `impose_prior=true`.
- `sigma_points`: sigma-point scheme; `"symmetric"` uses `2N_par+1` particles,
  `"simplex"` uses `N_par+2` particles.
"""
function Unscented(
    u0_mean::VV,
    uu0_cov::MM;
    α_reg::FT = 1.0,
    update_freq::IT = 0,
    modified_unscented_transform::Bool = true,
    impose_prior::Bool = false,
    prior_mean::Any = nothing,
    prior_cov::Any = nothing,
    sigma_points::String = "symmetric",
) where {FT <: AbstractFloat, IT <: Int, VV <: AbstractVector, MM <: AbstractMatrix}

    u0_mean = FT.(u0_mean)
    uu0_cov = FT.(uu0_cov)
    if impose_prior
        if isnothing(prior_mean)
            @info "`impose_prior=true` but `prior_mean=nothing`, taking initial mean as prior mean."
            prior_mean = u0_mean
        else
            prior_mean = FT.(prior_mean)
        end
        if isnothing(prior_cov)
            @info "`impose_prior=true` but `prior_cov=nothing`, taking initial covariance as prior covariance"
            prior_cov = uu0_cov
        else
            prior_cov = FT.(prior_cov)
        end
        α_reg = 1.0
        update_freq = 1
    end

    if sigma_points == "symmetric"
        N_ens = 2 * size(u0_mean, 1) + 1
    elseif sigma_points == "simplex"
        N_ens = size(u0_mean, 1) + 2
    else
        throw(ArgumentError("""
Unrecognized sigma_points type.

Expected:
    "symmetric" or "simplex"

Got:
    sigma_points = $(repr(sigma_points))
"""))
    end

    N_par = size(u0_mean, 1)
    # ensemble size

    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    if sigma_points == "symmetric"
        c_weights = zeros(FT, N_par)

        # set parameters λ, α
        α = min(sqrt(4 / N_par), 1.0)
        λ = α^2 * N_par - N_par

        c_weights[1:N_par] .= sqrt(N_par + λ)
        mean_weights[1] = λ / (N_par + λ)
        mean_weights[2:N_ens] .= 1 / (2 * (N_par + λ))
        cov_weights[1] = λ / (N_par + λ) + 1 - α^2 + 2.0
        cov_weights[2:N_ens] .= 1 / (2 * (N_par + λ))

    elseif sigma_points == "simplex"
        c_weights = zeros(FT, N_par, N_ens)

        # set parameters λ, α
        α = N_par / (4 * (N_par + 1))

        IM = zeros(FT, N_par, N_par + 1)
        IM[1, 1], IM[1, 2] = -1 / sqrt(2α), 1 / sqrt(2α)
        for i in 2:N_par
            for j in 1:i
                IM[i, j] = 1 / sqrt(α * i * (i + 1))
            end
            IM[i, i + 1] = -i / sqrt(α * i * (i + 1))
        end
        c_weights[:, 2:end] .= IM

        mean_weights .= 1 / (N_par + 1)
        mean_weights[1] = 0.0
        cov_weights .= α
        cov_weights[1] = 0.0

    end

    if modified_unscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end

    u_mean = Vector{FT}[]  # array of Vector{FT}'s
    push!(u_mean, u0_mean) # insert parameters at end of array (in this case just 1st entry)
    uu_cov = Matrix{FT}[]  # array of Matrix{FT}'s
    push!(uu_cov, uu0_cov) # insert parameters at end of array (in this case just 1st entry)

    obs_pred = Vector{FT}[]  # array of Vector{FT}'s

    Σ_ω = (2 - α_reg^2) * uu0_cov
    Σ_ν_scale = 2.0

    r = isnothing(prior_mean) ? u0_mean : prior_mean
    iter = 0

    Unscented(
        u_mean,
        uu_cov,
        obs_pred,
        c_weights,
        mean_weights,
        cov_weights,
        N_ens,
        Σ_ω,
        Σ_ν_scale,
        α_reg,
        r,
        update_freq,
        impose_prior,
        prior_mean,
        prior_cov,
        iter,
    )
end

"""
$(TYPEDSIGNATURES)

Construct an `Unscented` process from a `ParameterDistribution`, using its mean and
covariance (in unconstrained space) as initial state and as the default prior.
"""
function Unscented(prior::ParameterDistribution; kwargs...)

    u0_mean = isa(mean(prior), AbstractVector) ? Vector(mean(prior)) : [mean(prior)] # mean of unconstrained distribution
    uu0_cov = Matrix(cov(prior)) # cov of unconstrained distribution

    return Unscented(u0_mean, uu0_cov; prior_mean = u0_mean, prior_cov = uu0_cov, kwargs...)

end

# Constructors for TransformUnscented are based off the above, and so are placed here

"""
$(TYPEDSIGNATURES)

Construct a `TransformUnscented` process by copying all fields from an existing `Unscented`
process, adding an empty buffer required by the transform update.
"""
function TransformUnscented(process::UU) where {UU <: Unscented}
    return TransformUnscented(
        process.u_mean,
        process.uu_cov,
        process.obs_pred,
        process.c_weights,
        process.mean_weights,
        process.cov_weights,
        process.N_ens,
        process.Σ_ω,
        process.Σ_ν_scale,
        process.α_reg,
        process.r,
        process.update_freq,
        process.impose_prior,
        process.prior_mean,
        process.prior_cov,
        process.iter,
        [], # buffer
    )
end

"""
$(TYPEDSIGNATURES)

Construct a `TransformUnscented` process from an initial mean and covariance.

Accepts the same keyword arguments as `Unscented(u0_mean, uu0_cov; ...)`.
"""
function TransformUnscented(u0_mean::VV, uu0_cov::MM; kwargs...) where {VV <: AbstractVector, MM <: AbstractMatrix}
    process = Unscented(u0_mean, uu0_cov; kwargs...) # use UKI constructor
    return TransformUnscented(process)
end

"""
$(TYPEDSIGNATURES)

Construct a `TransformUnscented` process from a `ParameterDistribution`, using its mean
and covariance (in unconstrained space) as the initial state and default prior.
"""
function TransformUnscented(prior::ParameterDistribution; kwargs...)
    process = Unscented(prior; kwargs...) # use UKI constructor
    return TransformUnscented(process)
end

#

# Special constructor for UKI Object
"""
$(TYPEDSIGNATURES)

Construct an `EnsembleKalmanProcess` for an `Unscented` or `TransformUnscented` process
from an `ObservationSeries`.

The initial sigma ensemble is generated internally from the mean and covariance stored in
`process`; do not pass a pre-built ensemble.
"""
function EnsembleKalmanProcess(
    observation_series::OS,
    process::UorTU;
    kwargs...,
) where {UorTU <: Union{Unscented, TransformUnscented}, OS <: ObservationSeries}
    # use the distribution stored in process to generate initial ensemble
    init_params = update_ensemble_prediction!(process, 0.0)

    return EnsembleKalmanProcess(init_params, observation_series, process; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Construct an `EnsembleKalmanProcess` for an `Unscented` or `TransformUnscented` process
from a single `Observation`.

The initial sigma ensemble is generated internally from the mean and covariance stored in
`process`; do not pass a pre-built ensemble.
"""
function EnsembleKalmanProcess(
    observation::OB,
    process::UorTU;
    kwargs...,
) where {UorTU <: Union{Unscented, TransformUnscented}, OB <: Observation}

    observation_series = ObservationSeries(observation)
    return EnsembleKalmanProcess(observation_series, process; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Construct an `EnsembleKalmanProcess` for an `Unscented` or `TransformUnscented` process
from a raw observation mean vector and noise covariance.

The initial sigma ensemble is generated internally from the mean and covariance stored in
`process`; do not pass a pre-built ensemble.

# Arguments
- `obs_mean`: observed data vector.
- `obs_noise_cov`: observation noise covariance matrix or uniform scaling.
- `process`: an `Unscented` or `TransformUnscented` process.
"""
function EnsembleKalmanProcess(
    obs_mean::AbstractVector{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    process::UorTU;
    kwargs...,
) where {FT <: AbstractFloat, UorTU <: Union{Unscented, TransformUnscented}}

    observation = Observation(Dict("samples" => obs_mean, "covariances" => obs_noise_cov, "names" => "observation"))
    return EnsembleKalmanProcess(observation, process; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Return the prior mean stored in an `Unscented` or `TransformUnscented` process.
"""
get_prior_mean(process::UorTU) where {UorTU <: Union{Unscented, TransformUnscented}} = process.prior_mean

"""
$(TYPEDSIGNATURES)

Return the prior covariance stored in an `Unscented` or `TransformUnscented` process.
"""
get_prior_cov(process::UorTU) where {UorTU <: Union{Unscented, TransformUnscented}} = process.prior_cov


"""
$(TYPEDSIGNATURES)

Return the initial sigma ensemble for an `Unscented` or `TransformUnscented` process.

Parameters are returned as columns in unconstrained space by default; pass
`constrained=true` to transform to constrained space via `prior`.

Note: this function exists to inspect the initial sigma ensemble without constructing
an `EnsembleKalmanProcess` object. Do not pass the returned ensemble into
`EnsembleKalmanProcess` — the constructor generates it internally.
"""
function construct_initial_ensemble(
    prior::ParameterDistribution,
    process::UorTU;
    constrained = false,
) where {UorTU <: Union{Unscented, TransformUnscented}}
    u0_mean = process.u_mean[1]
    u0u0_cov = process.uu_cov[1]
    sigmas = construct_sigma_ensemble(process, u0_mean, u0u0_cov)
    if constrained
        return transform_unconstrained_to_constrained(prior, sigmas)
    else
        return sigmas
    end
end


"""
$(TYPEDSIGNATURES)

Return a `FailureHandler` for an `Unscented` process that ignores ensemble failures by
proceeding with the analysis and prediction steps on all particles, including failed ones.
"""
function FailureHandler(process::Unscented, method::IgnoreFailures)
    function failsafe_update(uki, u, g, u_idx, g_idx, failed_ens)
        #perform analysis on the model runs
        update_ensemble_analysis!(uki, u, g, u_idx, g_idx)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(get_process(uki), get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{Unscented, IgnoreFailures}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Return a `FailureHandler` for an `Unscented` process that handles failures by sampling
from the empirical Gaussian defined by successful sigma points.

The update rescales the mean weights and the off-center covariance weights of the
successful particles so that their sums match those of the original full weight sets.
"""
function FailureHandler(process::Unscented, method::SampleSuccGauss)
    function succ_gauss_analysis!(uki, u_p_full, g_full, u_idx, g_idx, failed_ens)
        process = get_process(uki)
        Σ_ν = process.Σ_ν_scale * get_obs_noise_cov(uki)[g_idx, g_idx]
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g_full, 2)))

        u_p = u_p_full[u_idx, :]
        obs_mean = get_obs(uki)[g_idx]
        g = g_full[g_idx, :]

        u_p_mean = construct_successful_mean(uki, u_p, successful_ens)
        uu_p_cov = construct_successful_cov(uki, u_p, u_p_mean, successful_ens)
        g_mean = construct_successful_mean(uki, g, successful_ens)
        gg_cov = construct_successful_cov(uki, g, g_mean, successful_ens) + Σ_ν / get_Δt(uki)[end]
        ug_cov = construct_successful_cov(uki, u_p, u_p_mean, g, g_mean, successful_ens)

        cov_est = [
            uu_p_cov ug_cov
            ug_cov' gg_cov
        ]

        # Localization
        FT = eltype(g_mean)
        cov_localized = get_localizer(uki).localize(cov_est, FT, size(u_p, 1), size(g, 1), size(u_p, 2))
        uu_p_cov, ug_cov, gg_cov = get_cov_blocks(cov_localized, size(u_p, 1))

        verbose = uki.verbose
        if process.impose_prior
            ug_cov_reg = [ug_cov uu_p_cov]
            gg_cov_reg = [gg_cov ug_cov'; ug_cov uu_p_cov+process.prior_cov[u_idx, u_idx] / get_Δt(uki)[end]]
            tmp = safe_linear_solve(gg_cov_reg', ug_cov_reg'; verbose)'
            u_mean = u_p_mean + tmp * [obs_mean - g_mean; process.prior_mean[u_idx] - u_p_mean]
            uu_cov = uu_p_cov - tmp * ug_cov_reg'

        else
            tmp = safe_linear_solve(gg_cov', ug_cov'; verbose)'
            u_mean = u_p_mean + tmp * (obs_mean - g_mean)
            uu_cov = uu_p_cov - tmp * ug_cov'

        end

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
    return FailureHandler{Unscented, SampleSuccGauss}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Return the sigma-point ensemble matrix of size `(N_x, N_ens)` built from mean `x_mean`
and covariance `x_cov`.

The layout and number of sigma points depend on the `sigma_points` scheme stored in
`process` (`"symmetric"` or `"simplex"`). When Cholesky factorization of `x_cov` fails,
an SVD-based fallback is used.
"""
function construct_sigma_ensemble(
    process::UorTU,
    x_mean::AbstractVector{FT},
    x_cov::AbstractMatrix{FT},
) where {FT <: AbstractFloat, UorTU <: Union{Unscented, TransformUnscented}}

    N_x = size(x_mean, 1)
    N_ens = process.N_ens
    c_weights = process.c_weights

    # compute cholesky factor L of x_cov
    local chol_xx_cov
    try
        chol_xx_cov = cholesky(Hermitian(x_cov)).L
    catch
        _, S, Ut = svd(x_cov)
        # find the first singular value that is smaller than 1e-8
        ind_0 = searchsortedfirst(S, 1e-8, rev = true)
        S[ind_0:end] .= S[ind_0 - 1]
        chol_xx_cov = (qr(sqrt.(S) .* Ut).R)'
    end

    x = zeros(FT, N_x, N_ens)
    x[:, 1] = x_mean

    if isa(c_weights, AbstractVector{FT})
        for i in 1:N_x
            x[:, i + 1] = x_mean + c_weights[i] * chol_xx_cov[:, i]
            x[:, i + 1 + N_x] = x_mean - c_weights[i] * chol_xx_cov[:, i]
        end
    elseif isa(c_weights, AbstractMatrix{FT})
        for i in 2:(N_x + 2)
            x[:, i] = x_mean + chol_xx_cov * c_weights[:, i]
        end
    end

    return x
end


"""
$(TYPEDSIGNATURES)

Return the weighted mean of an ensemble `x` using the UKI quadrature weights.

When `x` is a matrix, each column is treated as one ensemble member and the result
is a vector. When `x` is a vector, the result is a scalar.
"""
function construct_mean(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT};
    mean_weights = get_process(uki).mean_weights,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    if isa(x, AbstractMatrix{FT})
        size(x, 2) == length(mean_weights) || throw(DimensionMismatch("""
Ensemble size does not match the number of quadrature mean weights.

Expected:
    size(x, 2) == length(mean_weights)

Got:
    size(x, 2) = $(size(x, 2))
    length(mean_weights) = $(length(mean_weights))
"""))
        return Array((mean_weights' * x')')
    else
        length(mean_weights) == length(x) || throw(DimensionMismatch("""
Ensemble size does not match the number of quadrature mean weights.

Expected:
    length(x) == length(mean_weights)

Got:
    length(x) = $(length(x))
    length(mean_weights) = $(length(mean_weights))
"""))
        return mean_weights' * x
    end
end

"""
$(TYPEDSIGNATURES)

Return the weighted mean of ensemble `x` restricted to `successful_indices`, rescaling
the quadrature weights so that their sum over the surviving particles matches the
original total.

If the central (first) particle has failed under the modified unscented transform, the
mean reverts to the uniform average over all successful particles.
"""
function construct_successful_mean(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    mean_weights = deepcopy(get_process(uki).mean_weights)
    # Check if modified
    if sum(mean_weights[2:end]) ≈ 0 && !(1 in successful_indices)
        mean_weights .= 1 / length(successful_indices)
    else
        mean_weights = mean_weights ./ sum(mean_weights[successful_indices])
    end
    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_mean(uki, x_succ; mean_weights = mean_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

Return the weighted covariance of ensemble `x` with respect to mean `x_mean`.

When `x_mean` is `nothing` it is computed via `construct_mean`. When `x` is a matrix
the result is a matrix; when `x` is a vector the result is a scalar.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing} = nothing;
    cov_weights = get_process(uki).cov_weights,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    x_mean = isnothing(x_mean) ? construct_mean(uki, x) : x_mean

    if isa(x, AbstractMatrix{FT})
        isa(x_mean, AbstractVector{FT}) || throw(ArgumentError("""
When the ensemble is a matrix, the ensemble mean must be a vector.

Expected:
    x_mean::AbstractVector{$FT}

Got:
    typeof(x_mean) = $(typeof(x_mean))
    size(x) = $(size(x))

Suggestion:
    Pass x_mean = nothing to compute the mean automatically, or provide a vector of length $(size(x, 1)).
"""))
        N_x, N_ens = size(x)
        xx_cov = zeros(FT, N_x, N_x)

        for i in 1:N_ens
            xx_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (x[:, i] - x_mean)'
        end

        add_diagonal_regularization!(xx_cov)
    else
        isa(x_mean, FT) || throw(ArgumentError("""
When the ensemble is a vector, the ensemble mean must be a scalar.

Expected:
    x_mean::$FT

Got:
    typeof(x_mean) = $(typeof(x_mean))
    length(x) = $(length(x))

Suggestion:
    Pass x_mean = nothing to compute the mean automatically, or provide a scalar of type $FT.
"""))
        N_ens = length(x)
        xx_cov = FT(0)

        for i in 1:N_ens
            xx_cov += cov_weights[i] * (x[i] - x_mean) * (x[i] - x_mean)
        end
    end
    return xx_cov
end

"""
$(TYPEDSIGNATURES)

Return the weighted covariance of ensemble `x` restricted to `successful_indices`,
rescaling the off-center quadrature weights so that their sum matches the original.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    cov_weights = deepcopy(get_process(uki).cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_cov(uki, x_succ, x_mean; cov_weights = cov_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

Return the weighted cross-covariance between ensemble `x` (with mean `x_mean`) and
ensemble `obs_mean` (with mean `y_mean`).
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT};
    cov_weights = get_process(uki).cov_weights,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    N_x, N_ens = size(x)
    N_y = length(y_mean)
    xy_cov = zeros(FT, N_x, N_y)

    for i in 1:N_ens
        xy_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (obs_mean[:, i] - y_mean)'
    end

    return xy_cov
end

"""
$(TYPEDSIGNATURES)

Return the weighted cross-covariance between ensemble `x` (with mean `x_mean`) and
ensemble `obs_mean` (with mean `y_mean`) restricted to `successful_indices`, rescaling
the off-center quadrature weights so that their sum matches the original.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    N_ens, N_x, N_y = get_N_ens(uki), length(x_mean), length(y_mean)

    cov_weights = deepcopy(get_process(uki).cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    obs_mean_succ = isa(x, AbstractMatrix) ? obs_mean[:, successful_indices] : obs_mean[successful_indices]
    return construct_cov(uki, x_succ, x_mean, obs_mean_succ, y_mean; cov_weights = cov_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

Constructs weighted perturbations of `x` (a.k.a ensemble-square-root of the covariance of `x`) 
"""
function construct_perturbation(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing} = nothing;
    cov_weights = get_process(uki).cov_weights,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    x_mean = isnothing(x_mean) ? construct_mean(uki, x) : x_mean

    if isa(x, AbstractMatrix{FT})
        isa(x_mean, AbstractVector{FT}) || throw(ArgumentError("""
When the ensemble is a matrix, the ensemble mean must be a vector.

Expected:
    x_mean::AbstractVector{$FT}

Got:
    typeof(x_mean) = $(typeof(x_mean))
    size(x) = $(size(x))

Suggestion:
    Pass x_mean = nothing to compute the mean automatically, or provide a vector of length $(size(x, 1)).
"""))
        xx_pert = zeros(size(x))
        for i in 2:size(xx_pert, 2) # first column always zero (as it is the mean)
            xx_pert[:, i] = sqrt(cov_weights[i]) * (x[:, i] .- x_mean)
        end
    else
        isa(x_mean, FT) || throw(ArgumentError("""
When the ensemble is a vector, the ensemble mean must be a scalar.

Expected:
    x_mean::$FT

Got:
    typeof(x_mean) = $(typeof(x_mean))
    length(x) = $(length(x))

Suggestion:
    Pass x_mean = nothing to compute the mean automatically, or provide a scalar of type $FT.
"""))
        N_ens = length(x)
        xx_pert = zeros(N_ens)

        for i in 2:N_ens # first entry always zero (as it is the mean)
            xx_pert[i] += sqrt(cov_weights[i]) * (x[i] .- x_mean)
        end
    end
    return xx_pert
end

"""
$(TYPEDSIGNATURES)

Constructs weighted perturbations of `x` (a.k.a ensemble-square-root of the covariance of `x`) over successful particles by rescaling the off-center weights over the successful off-center particles.
"""
function construct_successful_perturbation(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}

    cov_weights = deepcopy(get_process(uki).cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_perturbation(uki, x_succ, x_mean; cov_weights = cov_weights[successful_indices])
end




"""
$(TYPEDSIGNATURES)

Perform the UKI prediction step for parameter indices `u_idx` and return the new sigma
ensemble as a matrix of size `(N_parameters, N_ens)`.

The predicted mean and covariance are propagated by the regularized evolution model
(`α_reg`, `r`, `Σ_ω`). If `update_freq > 0` and the current iteration is a multiple of
`update_freq`, the evolution covariance `Σ_ω` is updated from the current parameter
covariance.
"""
function update_ensemble_prediction!(
    process::UorTU,
    Δt::FT,
    u_idx::Vector{Int},
) where {FT <: AbstractFloat, UorTU <: Union{Unscented, TransformUnscented}}

    process.iter += 1
    # update evolution covariance matrix
    if process.update_freq > 0 && process.iter % process.update_freq == 0
        process.Σ_ω[u_idx, u_idx] = (2 - process.α_reg^2) * process.uu_cov[end][u_idx, u_idx]
    end

    u_mean = process.u_mean[end][u_idx]
    uu_cov = process.uu_cov[end][u_idx, u_idx]

    α_reg = process.α_reg
    r = process.r[u_idx]
    Σ_ω = process.Σ_ω[u_idx, u_idx]

    N_par = length(u_mean[1])
    ############# Prediction step:

    u_p_mean = α_reg * u_mean + (1 - α_reg) * r
    uu_p_cov = α_reg^2 * uu_cov + Σ_ω * Δt

    ############ Generate sigma points
    u_p = construct_sigma_ensemble(process, u_p_mean, uu_p_cov)
    return u_p
end

update_ensemble_prediction!(
    process::UorTU,
    Δt::FT,
) where {FT <: AbstractFloat, UorTU <: Union{Unscented, TransformUnscented}} =
    update_ensemble_prediction!(process, Δt, collect(1:length(process.u_mean[end])))

"""
$(TYPEDSIGNATURES)

Perform the UKI analysis step for an `Unscented` process, updating the stored parameter
mean, covariance, and predicted observations in-place.

`g_full` is the full `(N_obs, N_ens)` matrix of predicted observations; only the rows in
`g_idx` and the parameter dimensions in `u_idx` are updated. When `impose_prior=true`,
uses the augmented-system formulation.
"""
function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, U},
    u_p_full::AbstractMatrix{FT},
    g_full::AbstractMatrix{FT},
    u_idx::Vector{Int},
    g_idx::Vector{Int},
) where {FT <: AbstractFloat, IT <: Int, U <: Unscented}

    process = get_process(uki)
    Σ_ν = process.Σ_ν_scale * get_obs_noise_cov(uki)[g_idx, g_idx]

    u_p = u_p_full[u_idx, :]
    obs_mean = get_obs(uki)[g_idx]
    g = g_full[g_idx, :]

    u_p_mean = construct_mean(uki, u_p)
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)

    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν / get_Δt(uki)[end]
    ug_cov = construct_cov(uki, u_p, u_p_mean, g, g_mean)

    cov_est = [
        uu_p_cov ug_cov
        ug_cov' gg_cov
    ]
    # Localization
    cov_localized = get_localizer(uki).localize(cov_est, FT, size(u_p, 1), size(g, 1), size(u_p, 2))
    uu_p_cov, ug_cov, gg_cov = get_cov_blocks(cov_localized, size(u_p)[1])

    verbose = uki.verbose
    if process.impose_prior
        ug_cov_reg = [ug_cov uu_p_cov]
        gg_cov_reg = [
            gg_cov ug_cov'
            ug_cov uu_p_cov+process.prior_cov[u_idx, u_idx] / get_Δt(uki)[end]
        ]
        tmp = safe_linear_solve(gg_cov_reg', ug_cov_reg'; verbose)'
        u_mean = u_p_mean + tmp * [obs_mean - g_mean; process.prior_mean[u_idx] - u_p_mean]
        uu_cov = uu_p_cov - tmp * ug_cov_reg'
    else
        tmp = safe_linear_solve(gg_cov', ug_cov'; verbose)'
        u_mean = u_p_mean + tmp * (obs_mean - g_mean)
        uu_cov = uu_p_cov - tmp * ug_cov'
    end

    ########### Save results
    process.obs_pred[end][g_idx] .= g_mean
    process.u_mean[end][u_idx] .= u_mean
    process.uu_cov[end][u_idx, u_idx] .= uu_cov

end

function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, U},
    g::AbstractMatrix{FT},
    process::U,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    group_idx = 0,
    failed_ens = nothing,
    ekp_kwargs...,
) where {FT <: AbstractFloat, IT <: Int, U <: Unscented}
    #catch works when g_in non-square 
    u_p_old = get_u_final(uki)

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

"""
$(TYPEDSIGNATURES)

Return the mean unconstrained parameter vector at the given `iteration`.
"""
function get_u_mean(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    iteration::IT,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    return get_process(uki).u_mean[iteration]
end

"""
$(TYPEDSIGNATURES)

Return the unconstrained parameter covariance matrix at the given `iteration`.
"""
function get_u_cov(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
    iteration::IT,
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    return get_process(uki).uu_cov[iteration]
end


function compute_loss_at_mean(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    mean_g = get_process(uki).obs_pred[end]
    diff = get_obs(uki) - mean_g
    X = lmul_obs_noise_cov_inv(uki, diff)
    newerr = 1.0 / length(mean_g) * dot(diff, X)
    return newerr
end

function compute_unweighted_loss_at_mean(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    mean_g = get_process(uki).obs_pred[end]
    diff = get_obs(uki) - mean_g
    newerr = 1.0 / length(mean_g) * dot(diff, diff)
    return newerr
end

function compute_crps(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    g = get_g_final(uki)
    successful_ens, _ = split_indices_by_success(g)
    g_mean = construct_successful_mean(uki, g, successful_ens)
    diff = get_obs(uki) - g_mean
    if length(g_mean) > length(successful_ens) # dim > ens
        # get svd directly from the perturbations (not samples) 
        g_perturb = construct_successful_perturbation(uki, g, g_mean, successful_ens)[:, 2:end] # first column zeros
        g_svd = svd(g_perturb) # Note this svd gives, .S are sqrt-evals of cov-g
        white_diff = 1 ./ g_svd.S .* g_svd.U' * diff # as g_perturb was tall

        dist = Normal(0, 1)
        indep_crps = white_diff .* (2 .* cdf.(dist, white_diff) .- 1) .+ 2 * pdf.(dist, white_diff) .- 1 ./ sqrt(π)
        avg_crps = 1 ./ length(g_svd.S) * sum(g_svd.S .* indep_crps)
        return avg_crps

    else  # dim < ens
        g_cov = construct_successful_cov(uki, g, g_mean, successful_ens)
        g_svd = svd(g_cov) # Note this svd gives, .S are evals

        white_diff = 1 ./ sqrt.(g_svd.S) .* g_svd.Vt * diff # ~N(0,I)

        dist = Normal(0, 1)
        indep_crps = white_diff .* (2 .* cdf.(dist, white_diff) .- 1) .+ 2 * pdf.(dist, white_diff) .- 1 ./ sqrt(π)
        avg_crps = 1 ./ length(g_svd.S) * sum(sqrt.(g_svd.S) .* indep_crps)
        return avg_crps
    end

end

"""
$(TYPEDSIGNATURES)

Return the RMSE for an `Unscented` process evaluated at the mean, equal to `sqrt(compute_loss_at_mean(uki))`.

For Unscented processes, averaging RMSE at individual sigma points is not meaningful.
"""
function compute_average_rmse(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    return sqrt(compute_loss_at_mean(uki))
end

"""
$(TYPEDSIGNATURES)

Return the unweighted RMSE for an `Unscented` process evaluated at the mean, equal to `sqrt(compute_unweighted_loss_at_mean(uki))`.

For Unscented processes, averaging RMSE at individual sigma points is not meaningful.
"""
function compute_average_unweighted_rmse(
    uki::EnsembleKalmanProcess{FT, IT, UorTU},
) where {FT <: AbstractFloat, IT <: Int, UorTU <: Union{Unscented, TransformUnscented}}
    return sqrt(compute_unweighted_loss_at_mean(uki))
end




"""
$(TYPEDSIGNATURES)

Evaluate a 2-dimensional Gaussian density on a regular grid and return the grid arrays and density values.

# Arguments
- `u_mean`: mean of the 2-d Gaussian (length-2 vector).
- `uu_cov`: 2×2 covariance matrix.
- `Nx`: number of grid points along the first dimension.
- `Ny`: number of grid points along the second dimension.
- `xx`: optional custom grid along the first dimension; computed from `u_mean` and `uu_cov` if `nothing`.
- `yy`: optional custom grid along the second dimension; computed from `u_mean` and `uu_cov` if `nothing`.

Returns `(xx, yy, Z)` where `xx` and `yy` are the 1-d grid vectors and `Z` is the `Nx × Ny` density matrix.
"""
function Gaussian_2d(
    u_mean::AbstractVector{FT},
    uu_cov::AbstractMatrix{FT},
    Nx::IT,
    Ny::IT;
    xx = nothing,
    yy = nothing,
) where {FT <: AbstractFloat, IT <: Int}
    # 2d Gaussian plot
    u_range = [min(5 * sqrt(uu_cov[1, 1]), 5); min(5 * sqrt(uu_cov[2, 2]), 5)]

    if xx === nothing
        xx = Array(LinRange(u_mean[1] - u_range[1], u_mean[1] + u_range[1], Nx))
    end
    if yy == nothing
        yy = Array(LinRange(u_mean[2] - u_range[2], u_mean[2] + u_range[2], Ny))
    end
    X, Y = repeat(xx, 1, Ny), Array(repeat(yy, 1, Nx)')
    Z = zeros(FT, Nx, Ny)

    det_uu_cov = det(uu_cov)

    for ix in 1:Nx
        for iy in 1:Ny
            Δxy = [xx[ix] - u_mean[1]; yy[iy] - u_mean[2]]
            Z[ix, iy] = exp(-0.5 * (Δxy' / uu_cov * Δxy)) / (2 * pi * sqrt(det_uu_cov))
        end
    end

    return xx, yy, Z
end
