#Unscented Kalman Inversion: specific structures and function definitions

"""
    Unscented{FT<:AbstractFloat, IT<:Int} <: Process

An unscented Kalman Inversion process.
"""
mutable struct Unscented{FT <: AbstractFloat, IT <: Int} <: Process
    "an interable of arrays of size `N_parameters` containing the mean of the parameters (in each `uki` iteration a new array of mean is added)"
    u_mean  # ::Iterable{AbtractVector{FT}}
    "an iterable of arrays of size (`N_parameters x N_parameters`) containing the covariance of the parameters (in each `uki` iteration a new array of `cov` is added)"
    uu_cov  # ::Iterable{AbstractMatrix{FT}}
    "an iterable of arrays of size `N_y` containing the predicted observation (in each `uki` iteration a new array of predicted observation is added)"
    obs_pred # ::Iterable{AbstractVector{FT}}
    "weights in UKI"
    c_weights::AbstractVector{FT}
    mean_weights::AbstractVector{FT}
    cov_weights::AbstractVector{FT}
    "covariance of the artificial evolution error"
    Σ_ω::AbstractMatrix{FT}
    "covariance of the artificial observation error"
    Σ_ν_scale::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::AbstractVector{FT}
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
end



# outer constructors
function EnsembleKalmanProcess(
    obs_mean::AbstractVector{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    process::Unscented{FT, IT};
    Δt = FT(1),
    rng::AbstractRNG = Random.GLOBAL_RNG,
    failure_handler_method::FM = IgnoreFailures(),
) where {FT <: AbstractFloat, IT <: Int, FM <: FailureHandlingMethod}

    #initial parameters stored as columns
    init_params = [DataContainer(update_ensemble_prediction!(process, Δt), data_are_columns = true)]
    # Number of parameter
    N_u = length(process.u_mean[1])
    # ensemble size
    N_ens = 2N_u + 1 #stored with data as columns
    #store for model evaluations
    g = []
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])
    # failure handler
    fh = FailureHandler(process, failure_handler_method)

    EnsembleKalmanProcess{FT, IT, Unscented}(init_params, obs_mean, obs_noise_cov, N_ens, g, err, Δt, process, rng, fh)
end

function FailureHandler(process::Unscented, method::IgnoreFailures)
    function failsafe_update(uki, u, g, failed_ens)
        #perform analysis on the model runs
        update_ensemble_analysis!(uki, u, g)
        #perform new prediction output to model parameters u_p
        u = update_ensemble_prediction!(uki.process, uki.Δt[end])
        return u
    end
    return FailureHandler{Unscented, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::Unscented, method::SampleSuccGauss)

Provides a failsafe update that
 - computes all means and covariances over the successful sigma points,
 - rescales the mean weights and the off-center covariance weights of the
    successful particles to sum to the same value as the original weight sums.
"""
function FailureHandler(process::Unscented, method::SampleSuccGauss)
    function succ_gauss_analysis!(uki, u_p, g, failed_ens)

        obs_mean = uki.obs_mean
        Σ_ν = uki.process.Σ_ν_scale * uki.obs_noise_cov
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))

        ############# Prediction step
        u_p_mean = construct_successful_mean(uki, u_p, successful_ens)
        uu_p_cov = construct_successful_cov(uki, u_p, u_p_mean, successful_ens)

        ###########  Analysis step
        g_mean = construct_successful_mean(uki, g, successful_ens)
        gg_cov = construct_successful_cov(uki, g, g_mean, successful_ens) + Σ_ν / uki.Δt[end]
        ug_cov = construct_successful_cov(uki, u_p, u_p_mean, g, g_mean, successful_ens)

        tmp = ug_cov / gg_cov

        u_mean = u_p_mean + tmp * (obs_mean - g_mean)
        uu_cov = uu_p_cov - tmp * ug_cov'

        ########### Save results
        push!(uki.process.obs_pred, g_mean) # N_ens x N_data
        push!(uki.process.u_mean, u_mean) # N_ens x N_params
        push!(uki.process.uu_cov, uu_cov) # N_ens x N_data

        push!(uki.g, DataContainer(g, data_are_columns = true))

        compute_error!(uki)
    end
    function failsafe_update(uki, u, g, failed_ens)
        #perform analysis on the model runs
        succ_gauss_analysis!(uki, u, g, failed_ens)
        #perform new prediction output to model parameters u_p
        u = update_ensemble_prediction!(uki.process, uki.Δt[end])
        return u
    end
    return FailureHandler{Unscented, SampleSuccGauss}(failsafe_update)
end

"""
    Unscented(
        u0_mean::AbstractVector{FT},
        uu0_cov::AbstractMatrix{FT};
        α_reg::FT = 1.0,
        update_freq::IT = 1,
        modified_unscented_transform::Bool = true,
        prior_mean::Union{AbstractVector{FT}, Nothing} = nothing,
    ) where {FT <: AbstractFloat, IT <: Int}

Construct an Unscented Inversion EnsembleKalmanProcess.

Arguments
=========

  - `u0_mean`: Mean at initialization.
  - `uu0_cov`: Covariance at initialization.
  - `α_reg`: Hyperparameter controlling regularization toward the prior mean (0 < `α_reg` ≤ 1),
  default should be 1, without regulariazion.
  - `update_freq`: Set to 0 when the inverse problem is not identifiable, 
  namely the inverse problem has multiple solutions, the covariance matrix
  will represent only the sensitivity of the parameters, instead of
  posterior covariance information; set to 1 (or anything > 0) when
  the inverse problem is identifiable, and the covariance matrix will
  converge to a good approximation of the posterior covariance with an
  uninformative prior.
  - `modified_unscented_transform`: Modification of the UKI quadrature given
    in Huang et al (2021).
  - `prior_mean`: Prior mean used for regularization.
"""
function Unscented(
    u0_mean::AbstractVector{FT},
    uu0_cov::AbstractMatrix{FT};
    α_reg::FT = 1.0,
    update_freq::IT = 1,
    modified_unscented_transform::Bool = true,
    prior_mean::Union{AbstractVector{FT}, Nothing} = nothing,
) where {FT <: AbstractFloat, IT <: Int}

    N_u = size(u0_mean, 1)
    # ensemble size
    N_ens = 2 * N_u + 1

    c_weights = zeros(FT, N_u)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    # set parameters λ, α
    α = min(sqrt(4 / N_u), 1.0)
    λ = α^2 * N_u - N_u


    c_weights[1:N_u] .= sqrt(N_u + λ)
    mean_weights[1] = λ / (N_u + λ)
    mean_weights[2:N_ens] .= 1 / (2 * (N_u + λ))
    cov_weights[1] = λ / (N_u + λ) + 1 - α^2 + 2.0
    cov_weights[2:N_ens] .= 1 / (2 * (N_u + λ))

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
        Σ_ω,
        Σ_ν_scale,
        α_reg,
        r,
        update_freq,
        iter,
    )
end


"""
    construct_sigma_ensemble(
        process::Unscented,
        x_mean::Array{FT},
        x_cov::AbstractMatrix{FT},
    ) where {FT <: AbstractFloat, IT <: Int}

Construct the sigma ensemble based on the mean `x_mean` and covariance `x_cov`.
"""
function construct_sigma_ensemble(
    process::Unscented,
    x_mean::AbstractVector{FT},
    x_cov::AbstractMatrix{FT},
) where {FT <: AbstractFloat, IT <: Int}

    N_x = size(x_mean, 1)

    c_weights = process.c_weights

    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    x = zeros(FT, N_x, 2 * N_x + 1)
    x[:, 1] = x_mean
    for i in 1:N_x
        x[:, i + 1] = x_mean + c_weights[i] * chol_xx_cov[:, i]
        x[:, i + 1 + N_x] = x_mean - c_weights[i] * chol_xx_cov[:, i]
    end

    return x
end


"""
construct_mean `x_mean` from ensemble `x`.
"""
function construct_mean(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractVecOrMat{FT};
    mean_weights = uki.process.mean_weights,
) where {FT <: AbstractFloat, IT <: Int}

    if isa(x, AbstractMatrix{FT})
        @assert size(x, 2) == length(mean_weights)
        return Array((mean_weights' * x')')
    else
        @assert length(mean_weights) == length(x)
        return mean_weights' * x
    end
end

"""
    construct_successful_mean(
        uki::EnsembleKalmanProcess{FT, IT, Unscented},
        x::AbstractVecOrMat{FT},
        successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
    ) where {FT <: AbstractFloat, IT <: Int}

Constructs mean over successful particles by rescaling the quadrature
weights over the successful particles. If the central particle fails
in a modified unscented transform, the mean is computed as the
ensemble mean over all successful particles.
"""
function construct_successful_mean(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractVecOrMat{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int}

    mean_weights = deepcopy(uki.process.mean_weights)
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
construct_cov `xx_cov` from ensemble `x` and mean `x_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing} = nothing;
    cov_weights = uki.process.cov_weights,
) where {FT <: AbstractFloat, IT <: Int}

    x_mean = isnothing(x_mean) ? construct_mean(uki, x) : x_mean

    if isa(x, AbstractMatrix{FT})
        @assert isa(x_mean, AbstractVector{FT})
        N_x, N_ens = size(x)
        xx_cov = zeros(FT, N_x, N_x)

        for i in 1:N_ens
            xx_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (x[:, i] - x_mean)'
        end
    else
        @assert isa(x_mean, FT)
        N_ens = length(x)
        xx_cov = FT(0)

        for i in 1:N_ens
            xx_cov += cov_weights[i] * (x[i] - x_mean) * (x[i] - x_mean)
        end
    end
    return xx_cov
end

"""
    construct_successful_cov(
        uki::EnsembleKalmanProcess{FT, IT, Unscented},
        x::AbstractVecOrMat{FT},
        x_mean::Union{AbstractVector{FT}, FT},
        successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
    ) where {FT <: AbstractFloat, IT <: Int}

Constructs variance of `x` over successful particles by rescaling the
off-center weights over the successful off-center particles.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int}

    cov_weights = deepcopy(uki.process.cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_cov(uki, x_succ, x_mean; cov_weights = cov_weights[successful_indices])
end

"""
construct_cov `xy_cov` from ensemble x and mean `x_mean`, ensemble `obs_mean` and mean `y_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT};
    cov_weights = uki.process.cov_weights,
) where {FT <: AbstractFloat, IT <: Int, P <: Process}

    N_x, N_ens = size(x)
    N_y = length(y_mean)
    xy_cov = zeros(FT, N_x, N_y)

    for i in 1:N_ens
        xy_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (obs_mean[:, i] - y_mean)'
    end

    return xy_cov
end

"""
    construct_successful_cov(
        uki::EnsembleKalmanProcess{FT, IT, Unscented},
        x::AbstractMatrix{FT},
        x_mean::AbstractArray{FT},
        obs_mean::AbstractMatrix{FT},
        y_mean::AbstractArray{FT},
        successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
    ) where {FT <: AbstractFloat, IT <: Int}

Constructs covariance of `x` and `obs_mean - y_mean` over successful particles by rescaling
the off-center weights over the successful off-center particles.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int}
    N_ens, N_x, N_y = uki.N_ens, length(x_mean), length(y_mean)

    cov_weights = deepcopy(uki.process.cov_weights)

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
uki prediction step : generate sigma points
"""
function update_ensemble_prediction!(process::Unscented, Δt::FT) where {FT <: AbstractFloat}

    process.iter += 1
    # update evolution covariance matrix
    if process.update_freq > 0 && process.iter % process.update_freq == 0
        process.Σ_ω = (2 - process.α_reg^2) * process.uu_cov[end]
    end

    u_mean = process.u_mean[end]
    uu_cov = process.uu_cov[end]

    α_reg = process.α_reg
    r = process.r
    Σ_ω = process.Σ_ω

    N_u = length(process.u_mean[1])
    ############# Prediction step:

    u_p_mean = α_reg * u_mean + (1 - α_reg) * r
    uu_p_cov = α_reg^2 * uu_cov + Σ_ω * Δt

    ############ Generate sigma points
    u_p = construct_sigma_ensemble(process, u_p_mean, uu_p_cov)
    return u_p
end


"""
uki analysis step 
g is the predicted observations  `Ny x N_ens` matrix
"""
function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    u_p::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
) where {FT <: AbstractFloat, IT <: Int}

    obs_mean = uki.obs_mean
    Σ_ν = uki.process.Σ_ν_scale * uki.obs_noise_cov

    ############# Prediction step:

    u_p_mean = construct_mean(uki, u_p)
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)

    ###########  Analysis step

    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν / uki.Δt[end]
    ug_cov = construct_cov(uki, u_p, u_p_mean, g, g_mean)

    tmp = ug_cov / gg_cov

    u_mean = u_p_mean + tmp * (obs_mean - g_mean)
    uu_cov = uu_p_cov - tmp * ug_cov'

    ########### Save results
    push!(uki.process.obs_pred, g_mean) # N_ens x N_data
    push!(uki.process.u_mean, u_mean) # N_ens x N_params
    push!(uki.process.uu_cov, uu_cov) # N_ens x N_data

    push!(uki.g, DataContainer(g, data_are_columns = true))

    compute_error!(uki)
end

function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    g_in::AbstractMatrix{FT},
    Δt_new = nothing,
    failed_ens = nothing,
) where {FT <: AbstractFloat, IT <: Int}
    #catch works when g_in non-square 
    if !(size(g_in)[2] == uki.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end

    u_p_old = get_u_final(uki)

    set_Δt!(uki, Δt_new)
    fh = uki.failure_handler

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g_in)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u_p = fh.failsafe_update(uki, u_p_old, g_in, failed_ens)

    push!(uki.u, DataContainer(u_p, data_are_columns = true))

    return u_p
end


function get_u_mean_final(uki::EnsembleKalmanProcess{FT, IT, Unscented}) where {FT <: AbstractFloat, IT <: Int}
    return uki.process.u_mean[end]
end


function compute_error!(uki::EnsembleKalmanProcess{FT, IT, Unscented}) where {FT <: AbstractFloat, IT <: Int}
    mean_g = uki.process.obs_pred[end]
    diff = uki.obs_mean - mean_g
    X = uki.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(uki.err, newerr)
end

function get_error(uki::EnsembleKalmanProcess{FT, IT, Unscented}) where {FT <: AbstractFloat, IT <: Int}
    return uki.err
end


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
