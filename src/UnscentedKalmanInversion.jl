#Unscented Kalman Inversion: specific structures and function definitions

"""
    Unscented{FT<:AbstractFloat, IT<:Int} <: Process

An unscented Kalman Inversion process
"""
mutable struct Unscented{FT <: AbstractFloat, IT <: Int} <: Process
    "a vector of arrays of size `N_parameters` containing the mean of the parameters (in each `uki` iteration a new array of mean is added)"
    u_mean::Vector{Vector{FT}}
    "a vector of arrays of size (`N_parameters x N_parameters`) containing the covariance of the parameters (in each `uki` iteration a new array of `cov` is added)"
    uu_cov::Vector{Matrix{FT}}
    "a vector of arrays of size `N_y` containing the predicted observation (in each `uki` iteration a new array of predicted observation is added)"
    obs_pred::Vector{Vector{FT}}
    "weights in UKI"
    c_weights::Vector{FT}
    mean_weights::Vector{FT}
    cov_weights::Vector{FT}
    "covariance of the artificial evolution error"
    Σ_ω::Matrix{FT}
    "covariance of the artificial observation error"
    Σ_ν_scale::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::Vector{FT}
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
end



# outer constructors
function EnsembleKalmanProcess(
    obs_mean::Vector{FT},
    obs_noise_cov::Matrix{FT},
    process::Unscented{FT, IT};
    Δt = FT(1),
) where {FT <: AbstractFloat, IT <: Int}

    #initial parameters stored as columns
    init_params = [DataContainer(update_ensemble_prediction!(process), data_are_columns = true)]
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

    EnsembleKalmanProcess{FT, IT, Unscented}(init_params, obs_mean, obs_noise_cov, N_ens, g, err, Δt, process)
end



"""
    Unscented(
        u0_mean::Vector{FT},
        uu0_cov::Matrix{FT},
        α_reg::FT,
        update_freq::IT;
        modified_unscented_transform::Bool = true,
        prior_mean::Union{Vector{FT}, Nothing} = nothing,
        κ::FT = 0.0,
        β::FT = 2.0,
    ) where {FT <: AbstractFloat, IT <: Int}

Construct an Unscented Inversion EnsembleKalmanProcess.

Inputs:
 - u0_mean : Mean at initialization.
 - uu0_cov : Covariance at initialization.
 - α_reg : Hyperparameter controlling regularization toward the prior mean (0 < `α_reg` ≤ 1),
    default should be 1, without regulariazion.
 - update_freq : Set to 0 when the inverse problem is not identifiable, 
    namely the inverse problem has multiple solutions, the covariance matrix
    will represent only the sensitivity of the parameters, instead of
    posterior covariance information; set to 1 (or anything > 0) when
    the inverse problem is identifiable, and the covariance matrix will
    converge to a good approximation of the posterior covariance with an
    uninformative prior.
 - modified_unscented_transform : Modification of the UKI quadrature given
    in Huang et al (2021).
 - prior_mean : Prior mean used for regularization.
 - κ : 
 - β : 
                  
"""
function Unscented(
    u0_mean::Vector{FT},
    uu0_cov::Matrix{FT},
    α_reg::FT,
    update_freq::IT;
    modified_unscented_transform::Bool = true,
    prior_mean::Union{Vector{FT}, Nothing} = nothing,
    κ::FT = 0.0,
    β::FT = 2.0,
) where {FT <: AbstractFloat, IT <: Int}

    N_u = size(u0_mean, 1)
    # ensemble size
    N_ens = 2 * N_u + 1

    c_weights = zeros(FT, N_u)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    # todo parameters λ, α, β
    α = min(sqrt(4 / (N_u + κ)), 1.0)
    λ = α^2 * (N_u + κ) - N_u


    c_weights[1:N_u] .= sqrt(N_u + λ)
    mean_weights[1] = λ / (N_u + λ)
    mean_weights[2:N_ens] .= 1 / (2 * (N_u + λ))
    cov_weights[1] = λ / (N_u + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1 / (2 * (N_u + λ))

    if modified_unscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end

    u_mean = Vector{FT}[]  # array of Vector{FT}'s
    push!(u_mean, u0_mean) # insert parameters at end of array (in this case just 1st entry)
    uu_cov = Matrix{FT}[] # array of Matrix{FT}'s
    push!(uu_cov, uu0_cov) # insert parameters at end of array (in this case just 1st entry)

    obs_pred = Vector{FT}[]  # array of Vector{FT}'s
    err = FT[]

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
    function construct_sigma_ensemble(
        process::Unscented,
        x_mean::Array{FT},
        x_cov::Matrix{FT},
    ) where {FT <: AbstractFloat, IT <: Int}

Construct the sigma ensemble based on the mean `x_mean` and covariance `x_cov`.
"""
function construct_sigma_ensemble(
    process::Unscented,
    x_mean::Array{FT},
    x_cov::Matrix{FT},
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
    x::Matrix{FT},
) where {FT <: AbstractFloat, IT <: Int}
    N_x, N_ens = size(x)

    @assert(uki.N_ens == N_ens)

    x_mean = zeros(FT, N_x)

    mean_weights = uki.process.mean_weights

    for i in 1:N_ens
        x_mean += mean_weights[i] * x[:, i]
    end

    return x_mean
end

"""
construct_cov `xx_cov` from ensemble `x` and mean `x_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::Matrix{FT},
    x_mean::Array{FT},
) where {FT <: AbstractFloat, IT <: Int}
    N_ens, N_x = uki.N_ens, size(x_mean, 1)

    cov_weights = uki.process.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i in 1:N_ens
        xx_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (x[:, i] - x_mean)'
    end

    return xx_cov
end

"""
construct_cov `xy_cov` from ensemble x and mean `x_mean`, ensemble `obs_mean` and mean `y_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, Unscented},
    x::Matrix{FT},
    x_mean::Array{FT},
    obs_mean::Matrix{FT},
    y_mean::Array{FT},
) where {FT <: AbstractFloat, IT <: Int, P <: Process}
    N_ens, N_x, N_y = uki.N_ens, size(x_mean, 1), size(y_mean, 1)

    cov_weights = uki.process.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i in 1:N_ens
        xy_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (obs_mean[:, i] - y_mean)'
    end

    return xy_cov
end

"""
uki prediction step : generate sigma points
"""
function update_ensemble_prediction!(process::Unscented) where {FT <: AbstractFloat, IT <: Int}

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
    uu_p_cov = α_reg^2 * uu_cov + Σ_ω

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
    u_p::Matrix{FT},
    g::Matrix{FT},
) where {FT <: AbstractFloat, IT <: Int}

    obs_mean = uki.obs_mean
    Σ_ν = uki.process.Σ_ν_scale * uki.obs_noise_cov

    ############# Prediction step:

    u_p_mean = construct_mean(uki, u_p)
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)

    ###########  Analysis step

    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
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
    g_in::Matrix{FT},
) where {FT <: AbstractFloat, IT <: Int}
    #catch works when g_in non-square 
    if !(size(g_in)[2] == uki.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end

    u_p_old = get_u_final(uki)

    #perform analysis on the model runs
    update_ensemble_analysis!(uki, u_p_old, g_in)
    #perform new prediction output to model parameters u_p
    u_p = update_ensemble_prediction!(uki.process)

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
    u_mean::Vector{FT},
    uu_cov::Matrix{FT},
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
