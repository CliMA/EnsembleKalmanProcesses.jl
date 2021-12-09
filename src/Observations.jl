module Observations

using DocStringExtensions
using LinearAlgebra
using Statistics

export Obs

"""
    Obs{FT <: AbstractFloat}

Structure that contains the observations

# Fields

$(TYPEDFIELDS)
"""
struct Obs{FT <: AbstractFloat}
    "vector of observational samples, each of length sample_dim"
    samples::Vector{Vector{FT}}
    "covariance of the observational noise (assumed to be normally 
    distributed); `sample_dim x sample_dim` (where `sample_dim` is the number of 
    elements in each sample), or a scalar if the sample dim is 1. If not 
    supplied, `obs_noise_cov` is set to a diagonal matrix whose non-zero elements 
    are the variances of the samples, or to a scalar variance in the case of 
    1d samples. `obs_noise_cov` is set to nothing if only a single sample is 
    provided."
    obs_noise_cov::Union{Array{FT, 2}, FT, Nothing}
    "sample mean"
    mean::Union{Vector{FT}, FT}
    "names of the data"
    data_names::Union{Vector{String}, String}
end

# Constructors
function Obs(samples::Vector{Vector{FT}}, data_names::Union{Vector{String}, String}) where {FT <: AbstractFloat}

    N_samples = length(samples)
    # convert to sample_dim x N_samples to determine sample covariance

    if N_samples == 1
        # only one sample - this requires a bit more data massaging
        temp1 = convert(Array, reshape(hcat(samples...)', N_samples, :))
        temp = dropdims(temp1, dims = 1)
        samplemean = vec(mean(temp, dims = 2))
        obs_noise_cov = nothing
    else
        temp = convert(Array, reshape(hcat(samples...)', N_samples, :))
        if size(temp, 2) == 1
            # We have 1D samples, so the sample mean and covariance (which in
            # this case is actually the covariance) are scalars
            samplemean = mean(temp)
            obs_noise_cov = var(temp)
        else
            samplemean = vec(mean(temp, dims = 1))
            obs_noise_cov = cov(temp)
        end
    end
    Obs(samples, obs_noise_cov, samplemean, data_names)
end

function Obs(samples::Array{FT, 2}, data_names::Union{Vector{String}, String}) where {FT <: AbstractFloat}

    # samples is of size sample_dim x N_samples
    sample_dim, N_samples = size(samples)

    if N_samples == 1
        # Only one sample, so there is no covariance to be computed and the 
        # sample mean equals the sample itself
        obs_noise_cov = nothing
        samplemean = vec(samples)
        samples_vec = vec([vec(samples)])

    else
        # convert matrix of samples to a vector of vectors
        samples_vec = [samples[:, i] for i in 1:N_samples]
        if sample_dim == 1
            # We have 1D samples, so the sample mean and covariance (which in 
            # this case is actually the variance) are scalars
            samplemean = mean(samples)
            obs_noise_cov = var(samples)
        else
            samplemean = vec(mean(samples, dims = 2))
            obs_noise_cov = cov(samples, dims = 2)
        end
    end
    Obs(samples_vec, obs_noise_cov, samplemean, data_names)
end

function Obs(
    samples::Vector{Vector{FT}},
    obs_noise_cov::Array{FT, 2},
    data_names::Union{Vector{String}, String},
) where {FT <: AbstractFloat}

    N_samples = length(samples)
    sample_dim = length(samples[1])
    # convert to sample_dim x N_samples to determine sample covariance

    if N_samples == 1
        # only one sample - this requires a bit more data massaging
        temp1 = convert(Array, reshape(hcat(samples...)', N_samples, :))
        temp = dropdims(temp1, dims = 1)
        samplemean = vec(mean(temp, dims = 2))
        obsnoisecov = obs_noise_cov
    else
        temp = convert(Array, reshape(hcat(samples...)', N_samples, :))
        if sample_dim == 1
            # We have 1D samples, so the sample mean and covariance (which in
            # this case is actually the covariance) are scalars
            samplemean = mean(temp)
            err = ("When sample_dim is 1, obs_cov_noise must be a 1x1 matrix")
            @assert(size(obs_noise_cov) == (1, 1), err)
            obsnoisecov = obs_noise_cov[1]
        else
            samplemean = vec(mean(temp, dims = 1))
            err = ("obs_cov_noise must be of size sample_dim x sample_dim.
                   \tsample_dim: number of elements per observation sample")
            @assert(size(obs_noise_cov) == (sample_dim, sample_dim), err)
            obsnoisecov = obs_noise_cov
        end
    end


    Obs(samples, obsnoisecov, samplemean, data_names)
end

function Obs(
    samples::Array{FT, 2},
    obs_noise_cov::Union{Array{FT, 2}, Nothing},
    data_names::Union{Vector{String}, String},
) where {FT <: AbstractFloat}

    # samples is of size sample_dim x N_samples
    sample_dim, N_samples = size(samples)
    if N_samples == 1
        # sample mean equals the sample itself
        obsnoisecov = obs_noise_cov
        samplemean = vec(samples)
        samples_vec = vec([vec(samples)])
    else
        # convert matrix of samples to a vector of vectors
        samples_vec = [samples[:, i] for i in 1:N_samples]
        if sample_dim == 1
            # We have 1D samples, so the sample mean and covariance (which in 
            # this case is actually the variance) are scalars
            samplemean = mean(samples)
            err = ("When sample_dim is 1, obs_cov_noise must be a 1x1 matrix")
            @assert(size(obs_noise_cov) == (1, 1), err)
            obsnoisecov = obs_noise_cov[1]
        else
            samplemean = vec(mean(samples, dims = 2))
            err = ("obs_cov_noise must be of size sample_dim x sample_dim.
                   \tsample_dim: number of elements per observation sample")
            @assert(size(obs_noise_cov) == (sample_dim, sample_dim), err)
            obsnoisecov = obs_noise_cov
        end
    end

    Obs(samples_vec, obsnoisecov, samplemean, data_names)
end

end # module Observations
