module EnsembleKalmanProcessModule


using ..ParameterDistributionStorage
using ..DataStorage

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EnsembleKalmanProcess, Inversion, Sampler, Unscented
export MiniBatchKalmanProcess
export get_u, get_g, get_obs, get_obs_cov
export get_u_prior, get_u_final, get_u_mean_final, get_g_final
export get_N_iterations, get_error, get_obs_final, get_obs_cov_final
export construct_initial_ensemble
export compute_error!
export update_ensemble!
export find_ekp_stepsize
export Gaussian_2d


abstract type Process end

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

"""
    Sampler{FT<:AbstractFloat,IT<:Int} <: Process

An ensemble Kalman Sampler process
"""
struct Sampler{FT<:AbstractFloat} <: Process
  ""
  prior_mean::Vector{FT}
  ""
  prior_cov::Array{FT, 2}
end


"""
    Unscented{FT<:AbstractFloat, IT<:Int} <: Process

An unscented Kalman Inversion process
"""
mutable struct  Unscented{FT<:AbstractFloat, IT<:Int} <: Process
    "a vector of arrays of size N_parameters containing the mean of the parameters (in each uki iteration a new array of mean is added)"
    u_mean::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
    uu_cov::Vector{Array{FT, 2}}
    "a vector of arrays of size N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
    obs_pred::Vector{Array{FT, 1}}
    "weights in UKI"
    c_weights::Array{FT, 1}
    mean_weights::Array{FT, 1}
    cov_weights::Array{FT, 1}
    "covariance of the artificial evolution error"
    Σ_ω::Array{FT, 2}
    "covariance of the artificial observation error"
    Σ_ν_scale::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::Array{FT, 1}
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
end

abstract type KalmanProcessObject end

"""
    EnsembleKalmanProcess{FT<:AbstractFloat, IT<:Int}

Structure that is used in Ensemble Kalman processes

#Fields
$(DocStringExtensions.FIELDS)
"""
struct EnsembleKalmanProcess{FT<:AbstractFloat, IT<:Int, P<:Process} <: KalmanProcessObject
    "Array of stores for parameters (u), each of size [N_par × N_ens]"
    u::Array{DataContainer{FT}}
    "vector of the observed vector size [N_obs]"
    obs_mean::Vector{FT}
    "covariance matrix of the observational noise, of size [N_obs × N_obs]"
    obs_noise_cov::Array{FT, 2}
    "ensemble size"
    N_ens::IT
    "Array of stores for forward model outputs, each of size  [N_obs × N_ens]"
    g::Array{DataContainer{FT}}
    "vector of errors"
    err::Vector{FT}
    "vector of timesteps used in each EK iteration"
    Δt::Vector{FT}
    "the particular EK process (`Inversion` or `Sampler`)"
    process::P
end

# outer constructors
function EnsembleKalmanProcess(params::Array{FT, 2},
                               obs_mean,
                               obs_noise_cov::Array{FT, 2},
                               process::P;
                               Δt=FT(1)) where {FT<:AbstractFloat, P<:Process}

    #initial parameters stored as columns
    init_params=DataContainer(params, data_are_columns=true)
    # ensemble size
    N_ens = size(init_params,2) #stored with data as columns
    IT = typeof(N_ens)
    #store for model evaluations
    g=[] 
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    EnsembleKalmanProcess{FT, IT, P}([init_params], obs_mean, obs_noise_cov, N_ens, g,
                                     err, Δt, process)
end


struct MiniBatchKalmanProcess{FT<:AbstractFloat, IT<:Int, P<:Process} <: KalmanProcessObject
    "Array of stores for parameters (u), each of size [N_par × N_ens]"
    u::Array{DataContainer{FT}}
    "Array of stores for observations, each of size [N_obs_i]"
    obs_mean::Vector{Array{FT, 1}}
    "Array of stores for covariance matrix of the observational noise, each of size [N_obs_i × N_obs_i]"
    obs_noise_cov::Array{DataContainer{FT}}
    "ensemble size"
    N_ens::IT
    "Array of stores for forward model outputs, each of size  [N_obs × N_ens]"
    g::Array{DataContainer{FT}}
    "vector of errors"
    err::Vector{FT}
    "vector of timesteps used in each MBK iteration"
    Δt::Vector{FT}
    "the particular MBK process (`Inversion` or `Sampler`)"
    process::P
end

# outer constructors
function MiniBatchKalmanProcess(params::Array{FT, 2},
                               process::P;
                               Δt=FT(1)) where {FT<:AbstractFloat, P<:Process}

    #initial parameters stored as columns
    init_params=DataContainer(params, data_are_columns=true)
    # ensemble size
    N_ens = size(init_params,2) #stored with data as columns
    IT = typeof(N_ens)
    #store for model evaluations
    g=[]
    # store for observations
    obs_mean = []
    obs_noise_cov = []
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    MiniBatchKalmanProcess{FT, IT, P}([init_params], obs_mean, obs_noise_cov, N_ens, g,
                                     err, Δt, process)
end


"""
    get_u(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}

Get for the EKI iteration. Returns a DataContainer object unless array is specified.
"""
function get_u(ekp::KalmanProcessObject, iteration::IT; return_array=true) where {IT <: Integer}
    return  return_array ? get_data(ekp.u[iteration]) : ekp.u[iteration]
end

"""
    get_g(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}

Get for the EKI iteration. Returns a DataContainer object unless array is specified.
"""
function get_g(ekp::KalmanProcessObject, iteration::IT; return_array=true) where {IT <: Integer}
    return return_array ? get_data(ekp.g[iteration]) : ekp.g[iteration]
end

"""
    get_u(ekp::EnsembleKalmanProcess; return_array=true)

Get for the EKI iteration. Returns a DataContainer object unless array is specified.
"""
function get_u(ekp::KalmanProcessObject; return_array=true) where {IT <: Integer}
    N_stored_u = get_N_iterations(ekp)+1
    return [get_u(ekp, it, return_array=return_array) for it in 1:N_stored_u]
end

"""
    get_g(ekp::EnsembleKalmanProcess; return_array=true)

Get for the EKI iteration. Returns a DataContainer object unless array is specified.
"""
function get_g(ekp::KalmanProcessObject; return_array=true) where {IT <: Integer}
    N_stored_g = get_N_iterations(ekp)
    return [get_g(ekp, it, return_array=return_array) for it in 1:N_stored_g]
end


"""
    get_u_final(ekp::EnsembleKalmanProcess, return_array=true)

Get the final or prior iteration of parameters or model ouputs, returns a DataContainer Object if return_array is false.
"""
function get_u_final(ekp::KalmanProcessObject; return_array=true)
    return return_array ? get_u(ekp,size(ekp.u)[1]) : ekp.u[end]
end

"""
    get_u_prior(ekp::EnsembleKalmanProcess, return_array=true)

Get the final or prior iteration of parameters or model ouputs, returns a DataContainer Object if return_array is false.
"""

function get_u_prior(ekp::KalmanProcessObject; return_array=true)
    return return_array ? get_u(ekp,1) : ekp.u[1]
end

"""
    get_g_final(ekp::EnsembleKalmanProcess, return_array=true)

Get the final or prior iteration of parameters or model ouputs, returns a DataContainer Object if return_array is false.
"""

function get_g_final(ekp::KalmanProcessObject; return_array=true)
    return return_array ? get_g(ekp,size(ekp.g)[1]) : ekp.g[end]
end


function get_obs(ekp::MiniBatchKalmanProcess, iteration::IT) where {IT <: Integer}
    return ekp.obs_mean[iteration]
end

function get_obs(ekp::MiniBatchKalmanProcess; return_array=true) where {IT <: Integer}
    N_stored_obs = get_N_iterations(ekp)
    return [get_obs(ekp, it) for it in 1:N_stored_obs]
end

function get_obs_final(ekp::MiniBatchKalmanProcess)
    return ekp.obs_mean[end]
end

function get_obs_cov(ekp::MiniBatchKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}
    return return_array ? get_data(ekp.obs_noise_cov[iteration]) : ekp.obs_noise_cov[iteration]
end

function get_obs_cov(ekp::MiniBatchKalmanProcess; return_array=true) where {IT <: Integer}
    N_stored_obs = get_N_iterations(ekp)
    return [get_obs_cov(ekp, it, return_array=return_array) for it in 1:N_stored_obs]
end

function get_obs_cov_final(ekp::MiniBatchKalmanProcess; return_array=true)
    return return_array ? get_obs_cov(ekp,size(ekp.obs_noise_cov)[1]) : ekp.obs_noise_cov[end]
end

"""
    get_N_iterations(ekp::EnsembleKalmanProcess

get number of times update has been called (equals size(g), or size(u)-1) 
"""
function get_N_iterations(ekp::KalmanProcessObject)
    return size(ekp.u)[1] - 1 
end

"""
    construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distribution. Returned with parameters as columns
"""
function construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    parameters = sample_distribution(prior, N_ens) #of size [dim(param space) N_ens]
    return parameters
end

function compute_error!(ekp::EnsembleKalmanProcess)
    mean_g = dropdims(mean(get_g_final(ekp), dims=2), dims=2)
    diff = ekp.obs_mean - mean_g
    X = ekp.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end

function compute_error!(ekp::MiniBatchKalmanProcess)
    mean_g = dropdims(mean(get_g_final(ekp), dims=2), dims=2)
    obs_mean_ = get_obs_final(ekp)
    obs_cov_ = get_obs_cov_final(ekp)
    diff = obs_mean_ - mean_g
    X = obs_cov_ \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end

function get_error(ekp::KalmanProcessObject)
    return ekp.err
end

"""
   find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}

Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT,2}; cov_threshold::FT=0.01) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims=2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new=Δt)
        cov_new = cov(get_u_final(ekp_copy), dims=2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt/2
        end
    end

    return Δt

end

"""
    update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, <:Process}, g::Array{FT,2} cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}

Updates the ensemble according to which type of Process we have. Model outputs g need to be a N_obs × N_ens array (i.e data are columms)
"""
function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Inversion},
                          g::Array{FT,2};
                          cov_threshold::FT=0.01,
                          Δt_new=nothing,
                          deterministic_forward_map=true,) where {FT, IT}

    # Update follows eqns. (4) and (5) of Schillings and Stuart (2017)
    
    #catch works when g non-square
    if !(size(g)[2] == ekp.N_ens) 
         throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g do not match, try transposing g or check ensemble size"))
    end

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)
    N_obs = size(g, 1)

    cov_init = cov(u, dims=2)
    cov_ug = cov(u,g, dims = 2, corrected=false) # [N_par × N_obs]
    cov_gg = cov(g,g, dims = 2, corrected=false) # [N_obs × N_obs]

    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = (cov_gg + scaled_obs_noise_cov) \ (y - g)
    u += (cov_ug * tmp) # [N_par × N_ens]

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns=true))
    push!(ekp.g, DataContainer(g, data_are_columns=true))
    
    # Store error
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims=2)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string("New ensemble covariance determinant is less than ",
                     cov_threshold, " times its previous value.",
                     "\nConsider reducing the EK time step.")
    end
end

function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}}, g_in::Array{FT,2}) where {FT, IT}

    #catch works when g_in non-square 
    if !(size(g_in)[2] == ekp.N_ens) 
         throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end

    # u: N_ens × N_par
    # g: N_ens × N_obs
    u_old = get_u_final(ekp)
    u_old = permutedims(u_old,(2,1))
    u = u_old
    g = permutedims(g_in, (2,1))
   
    # u_mean: N_par × 1
    u_mean = mean(u', dims=2)
    # g_mean: N_obs × 1
    g_mean = mean(g', dims=2)
    # g_cov: N_obs × N_obs
    g_cov = cov(g, corrected=false)
    # u_cov: N_par × N_par
    u_cov = cov(u, corrected=false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- ekp.obs_mean
    # D: N_ens × N_ens
    D = (1/ekp.N_ens) * (E' * (ekp.obs_noise_cov \ R))

    Δt = 1/(norm(D) + 1e-8)

    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov' \ u_cov')') \
                  (u'
                    .- Δt * ( u' .- u_mean) * D
                    .+ Δt * u_cov * (ekp.process.prior_cov \ ekp.process.prior_mean)
                  )

    u = implicit' + sqrt(2*Δt) * rand(noise, ekp.N_ens)'

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns=false))
    push!(ekp.g, DataContainer(g, data_are_columns=false))
    # u_old is N_ens × N_par, g is N_ens × N_obs,
    # but stored in data container with N_ens as the 2nd dim

    compute_error!(ekp)

end






# outer constructors
function EnsembleKalmanProcess(
    obs_mean::Array{FT, 1},
    obs_noise_cov::Array{FT, 2},
    process::Unscented{FT, IT};
    Δt=FT(1)) where {FT<:AbstractFloat, IT<:Int}

    #initial parameters stored as columns
    init_params = [DataContainer(update_ensemble_prediction!(process), data_are_columns=true)]
    # Number of parameter
    N_u = length(process.u_mean[1])
    # ensemble size
    N_ens = 2*N_u + 1 #stored with data as columns
    #store for model evaluations
    g=[]
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    EnsembleKalmanProcess{FT, IT, Unscented}(init_params, obs_mean, obs_noise_cov, N_ens, g, err, Δt, process)
end

function MiniBatchKalmanProcess(
    process::Unscented{FT, IT};
    Δt=FT(1)) where {FT<:AbstractFloat, IT<:Int}

    #initial parameters stored as columns
    init_params = [DataContainer(update_ensemble_prediction!(process), data_are_columns=true)]
    # Number of parameter
    N_u = length(process.u_mean[1])
    # ensemble size
    N_ens = 2*N_u + 1 #stored with data as columns
    #store for model evaluations
    g=[]
    # store for observations
    obs_mean = []
    obs_noise_cov = []
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    MiniBatchKalmanProcess{FT, IT, Unscented}(init_params, obs_mean, obs_noise_cov, N_ens, g, err, Δt, process)
end

"""
EnsembleKalmanProcess Constructor 
u0_mean::Array{FT} : prior mean
uu0_cov::Array{FT, 2} : prior covariance
obs_mean::Array{FT,1} : observation 
obs_noise_cov::Array{FT, 2} : observation error covariance
α_reg::FT : regularization parameter toward u0 (0 < α_reg <= 1), default should be 1, without regulariazion
update_freq::IT : set to 0 when the inverse problem is not identifiable, 
                  namely the inverse problem has multiple solutions, 
                  the covariance matrix will represent only the sensitivity of the parameters, 
                  instead of posterior covariance information;
                  set to 1 (or anything > 0) when the inverse problem is identifiable, and 
                  the covariance matrix will converge to a good approximation of the 
                  posterior covariance with an uninformative prior.
                  
"""
function Unscented(
    u0_mean::Array{FT, 1}, 
    uu0_cov::Array{FT, 2},
    α_reg::FT,
    update_freq::IT;
    modified_unscented_transform::Bool = true,
    κ::FT = 0.0,
    β::FT = 2.0) where {FT<:AbstractFloat, IT<:Int}
    
    N_u = size(u0_mean,1)
    # ensemble size
    N_ens = 2*N_u+1
    
    
    c_weights = zeros(FT, N_u)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)
    
    # todo parameters λ, α, β
    
    α = min(sqrt(4/(N_u + κ)), 1.0)
    λ = α^2*(N_u + κ) - N_u
    
    
    c_weights[1:N_u]     .=  sqrt(N_u + λ)
    mean_weights[1] = λ/(N_u + λ)
    mean_weights[2:N_ens] .= 1/(2(N_u + λ))
    cov_weights[1] = λ/(N_u + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_u + λ))
    
    if modified_unscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end
    
    
    
    u_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(u_mean, u0_mean) # insert parameters at end of array (in this case just 1st entry)
    uu_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(uu_cov, uu0_cov) # insert parameters at end of array (in this case just 1st entry)
    
    obs_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
    err = FT[]
    
    Σ_ω = (2 - α_reg^2)*uu0_cov
    Σ_ν_scale = 2.0

    r = u0_mean
    iter = 0

    Unscented(u_mean, uu_cov,  obs_pred, c_weights, mean_weights, cov_weights, Σ_ω, Σ_ν_scale, α_reg, r, update_freq, iter)
end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(process::Unscented, x_mean::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}

    N_x = size(x_mean,1)
    
    c_weights = process.c_weights
    
    
    chol_xx_cov = cholesky(Hermitian(x_cov)).L
    
    x = zeros(FT, N_x, 2*N_x+1)
    x[:, 1] = x_mean
    for i = 1: N_x
        x[:, i+1] = x_mean + c_weights[i]*chol_xx_cov[:,i]
        x[:, i+1+N_x] = x_mean - c_weights[i]*chol_xx_cov[:,i]
    end
    
    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(uki::Union{EnsembleKalmanProcess{FT, IT,Unscented}, MiniBatchKalmanProcess{FT, IT,Unscented}},
    x::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_x, N_ens = size(x)
    
    @assert(uki.N_ens == N_ens)
    
    x_mean = zeros(FT, N_x)
    
    mean_weights = uki.process.mean_weights
    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[:, i]
    end
    
    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(uki::Union{EnsembleKalmanProcess{FT, IT,Unscented}, MiniBatchKalmanProcess{FT, IT,Unscented}},
    x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = uki.N_ens, size(x_mean,1)
    
    cov_weights = uki.process.cov_weights
    
    xx_cov = zeros(FT, N_x, N_x)
    
    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[:,i] - x_mean)*(x[:,i] - x_mean)'
    end
    
    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble obs_mean and mean y_mean
"""
function construct_cov(uki::Union{EnsembleKalmanProcess{FT, IT,Unscented}, MiniBatchKalmanProcess{FT, IT,Unscented}}, x::Array{FT,2}, x_mean::Array{FT}, obs_mean::Array{FT,2}, y_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int, P<:Process}
    N_ens, N_x, N_y = uki.N_ens, size(x_mean,1), size(y_mean,1)
    
    cov_weights = uki.process.cov_weights
    
    xy_cov = zeros(FT, N_x, N_y)
    
    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[:, i] - x_mean)*(obs_mean[:, i] - y_mean)'
    end
    
    return xy_cov
end

"""
uki prediction step : generate sigma points
"""
function update_ensemble_prediction!(process::Unscented) where {FT<:AbstractFloat, IT<:Int}
    
    process.iter += 1
    # update evolution covariance matrix
    if process.update_freq > 0 && process.iter%process.update_freq == 0
        process.Σ_ω = (2 - process.α_reg^2)process.uu_cov[end]
    end
    
    u_mean  = process.u_mean[end]
    uu_cov = process.uu_cov[end]
    
    α_reg = process.α_reg
    r = process.r
    Σ_ω = process.Σ_ω
    
    N_u = length(process.u_mean[1])
    ############# Prediction step:
    
    u_p_mean  = α_reg*u_mean + (1-α_reg)*r
    uu_p_cov = α_reg^2*uu_cov + Σ_ω
    
    ############ Generate sigma points
    u_p = construct_sigma_ensemble(process, u_p_mean, uu_p_cov)
    return u_p
end


"""
uki analysis step 
g is the predicted observations  Ny  by N_ens matrix
"""
function update_ensemble_analysis!(uki::EnsembleKalmanProcess{FT, IT,Unscented}, u_p::Array{FT, 2}, g::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    
    obs_mean = uki.obs_mean
    Σ_ν = uki.process.Σ_ν_scale * uki.obs_noise_cov
    
    N_u, N_y, N_ens = length(uki.process.u_mean[1]), length(uki.obs_mean), uki.N_ens
    ############# Prediction step:
    
    u_p_mean = construct_mean(uki, u_p) 
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)
    
    ###########  Analysis step
    
    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    ug_cov = construct_cov(uki, u_p, u_p_mean, g, g_mean)
    
    tmp = ug_cov/gg_cov
    
    u_mean =  u_p_mean + tmp*(obs_mean - g_mean)
    uu_cov =  uu_p_cov - tmp*ug_cov' 
    
    
    ########### Save resutls
    push!(uki.process.obs_pred, g_mean) # N_ens x N_data
    push!(uki.process.u_mean, u_mean) # N_ens x N_params
    push!(uki.process.uu_cov, uu_cov) # N_ens x N_data
    
    push!(uki.g, DataContainer(g, data_are_columns=true))

    compute_error!(uki)
end

"""
uki analysis step 
g is the predicted observations  Ny  by N_ens matrix
"""
function update_ensemble_analysis!(uki::MiniBatchKalmanProcess{FT, IT,Unscented}, u_p::Array{FT, 2}, g::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    
    obs_mean = get_obs_final(uki)
    Σ_ν = uki.process.Σ_ν_scale * get_obs_cov_final(uki)
    
    N_u, N_y, N_ens = length(uki.process.u_mean[1]), length(obs_mean), uki.N_ens
    ############# Prediction step:
    
    u_p_mean = construct_mean(uki, u_p) 
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)
    
    ###########  Analysis step
    
    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    ug_cov = construct_cov(uki, u_p, u_p_mean, g, g_mean)
    
    tmp = ug_cov/gg_cov
    
    u_mean =  u_p_mean + tmp*(obs_mean - g_mean)
    uu_cov =  uu_p_cov - tmp*ug_cov' 
    
    
    ########### Save resutls
    push!(uki.process.obs_pred, g_mean) # N_ens x N_data
    push!(uki.process.u_mean, u_mean) # N_ens x N_params
    push!(uki.process.uu_cov, uu_cov) # N_ens x N_data
    
    push!(uki.g, DataContainer(g, data_are_columns=true))

    compute_error!(uki)
end

function update_ensemble!(uki::EnsembleKalmanProcess{FT, IT, Unscented}, g_in::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    #catch works when g_in non-square 
    if !(size(g_in)[2] == uki.N_ens) 
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end
    
    u_p_old = get_u_final(uki)

    #perform analysis on the model runs
    update_ensemble_analysis!(uki, u_p_old, g_in)
    #perform new prediction output to model parameters u_p
    u_p = update_ensemble_prediction!(uki.process) 

    push!(uki.u, DataContainer(u_p, data_are_columns=true))

    return u_p

end

function update_ensemble!(uki::MiniBatchKalmanProcess{FT, IT, Unscented}, g_in::Array{FT, 2}, obs_mean_in::Array{FT}, obs_cov_in::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    #catch works when g_in non-square
    if !(size(g_in)[2] == uki.N_ens) 
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end
    
    u_p_old = get_u_final(uki)

    #perform analysis on the model runs
    update_observations!(uki, obs_mean_in, obs_cov_in)
    update_ensemble_analysis!(uki, u_p_old, g_in)
    #perform new prediction output to model parameters u_p
    u_p = update_ensemble_prediction!(uki.process) 

    push!(uki.u, DataContainer(u_p, data_are_columns=true))

    return u_p

end

function update_observations!(ekp::MiniBatchKalmanProcess{FT, IT, P}, obs_mean_in::Array{FT}, obs_cov_in::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int, P<:Process}
    
    push!(ekp.obs_mean, obs_mean_in)
    push!(ekp.obs_noise_cov, DataContainer(obs_cov_in, data_are_columns=true))

    return
end

function get_u_mean_final(uki::Union{EnsembleKalmanProcess{FT, IT,Unscented}, MiniBatchKalmanProcess{FT, IT,Unscented}}) where {FT<:AbstractFloat, IT<:Int}
    return uki.process.u_mean[end]
end


function compute_error!(uki::EnsembleKalmanProcess{FT, IT,Unscented}) where {FT<:AbstractFloat, IT<:Int}
    mean_g = uki.process.obs_pred[end]
    diff = uki.obs_mean - mean_g
    X = uki.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(uki.err, newerr)
end

function compute_error!(uki::MiniBatchKalmanProcess{FT, IT,Unscented}) where {FT<:AbstractFloat, IT<:Int}
    mean_g = uki.process.obs_pred[end]
    diff = get_obs_final(uki) - mean_g
    X = get_obs_cov_final(uki) \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(uki.err, newerr)
end


function Gaussian_2d(u_mean::Array{FT,1}, uu_cov::Array{FT,2}, Nx::IT, Ny::IT; xx = nothing, yy = nothing) where {FT<:AbstractFloat, IT<:Int}
    # 2d Gaussian plot
    u_range = [min(5*sqrt(uu_cov[1,1]), 5); min(5*sqrt(uu_cov[2,2]), 5)]

    if xx === nothing
        xx = Array(LinRange(u_mean[1] - u_range[1], u_mean[1] + u_range[1], Nx))
    end
    if yy == nothing
        yy = Array(LinRange(u_mean[2] - u_range[2], u_mean[2] + u_range[2], Ny))
    end
    X,Y = repeat(xx, 1, Ny), Array(repeat(yy, 1, Nx)')
    Z = zeros(FT, Nx, Ny)
    
    det_uu_cov = det(uu_cov)

    for ix = 1:Nx
        for iy = 1:Ny
            Δxy = [xx[ix] - u_mean[1]; yy[iy] - u_mean[2]]
            Z[ix, iy] = exp(-0.5*(Δxy'/uu_cov*Δxy)) / (2 * pi * sqrt(det_uu_cov))
        end
    end
    
    return xx, yy, Z
    
end
end # module
