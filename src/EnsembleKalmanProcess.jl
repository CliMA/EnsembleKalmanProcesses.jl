module EnsembleKalmanProcessModule


using ..ParameterDistributionStorage
using ..DataStorage

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EnsembleKalmanProcess 
export get_u, get_g
export get_u_prior, get_u_final, get_u_mean_final, get_g_final, get_N_iterations, get_error
export compute_error!
export update_ensemble!


abstract type Process end
#specific Processes and their exports are included after the general definitions

## begin general constructor and function definitions


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
    "the particular EK process (`Inversion` or `Sampler` or `Unscented`)"
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


## include the different types of Processes and their exports:

# struct Inversion
export Inversion
export find_ekp_stepsize
include("EnsembleKalmanInversion.jl")


# struct Sampler
export Sampler 
include("EnsembleKalmanSampler.jl")


# struct Unscented
export Unscented
export construct_initial_ensemble
export Gaussian_2d
include("UnscentedKalmanInversion.jl") 







end # module
