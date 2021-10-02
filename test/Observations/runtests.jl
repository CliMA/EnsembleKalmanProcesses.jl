using Test
using Random
using Statistics

using EnsembleKalmanProcesses.Observations

@testset "Observations" begin

    # Generate samples as vector of vectors
    sample_dim = 3  # sample dimension, i.e., number of elements per sample
    n_samples = 5   # number of samples
    samples = [vec(i * ones(sample_dim)) for i in 1:n_samples]
    data_names = ["d$(string(i))" for i in 1:sample_dim]
    obs = Obs(samples, data_names)
    @test obs.data_names == data_names
    @test obs.mean == [3.0, 3.0, 3.0]
    @test obs.obs_noise_cov == 2.5 * ones(3, 3)

    # Generate samples as vector of vectors, pass a covariance to Obs
    covar = ones(sample_dim, sample_dim)
    obs = Obs(samples, covar, data_names)
    @test obs.obs_noise_cov == covar
    covar_wrong_dims = ones(sample_dim, sample_dim - 1)
    @test_throws AssertionError Obs(samples, covar_wrong_dims, data_names)

    # Generate samples as a 2d-array (each row corresponds to 1 sample)
    samples_t = vcat([i * ones(sample_dim)' for i in 1:n_samples]...)
    samples = permutedims(samples_t, (2, 1))
    obs = Obs(samples, data_names)
    @test obs.mean == [3.0, 3.0, 3.0]
    @test obs.obs_noise_cov == 2.5 * ones(3, 3)

    # Generate samples as a 2d-array (each row corresponds to 1 sample), 
    # pass a covariance to Obs
    obs = Obs(samples, covar, data_names)
    @test obs.obs_noise_cov == covar
    @test_throws AssertionError Obs(samples, covar_wrong_dims, data_names)
    @test_throws AssertionError Obs(samples_t, covar, data_names) #as default is data are columns

    ## Edge cases:
    # Generating a single sample (a column vector)
    sample = reshape([1.0, 2.0, 3.0], 3, 1)
    data_name = "d1"
    obs = Obs(sample, data_name)
    @test obs.mean == vec(sample)
    @test obs.obs_noise_cov == nothing

    #Generating a single sample (vector of vector)
    sample = [[1.0, 2.0, 3.0]]
    data_name = "d1"
    obs = Obs(sample, data_name)
    @test obs.mean == sample[1]
    @test obs.obs_noise_cov == nothing

    # Generating a single sample with cov (a column vector)
    sample = reshape([1.0, 2.0, 3.0], 3, 1)
    obs_noise_cov = ones(3, 3)
    data_name = "d1"
    obs = Obs(sample, obs_noise_cov, data_name)
    @test obs.mean == vec(sample)
    @test obs.obs_noise_cov == obs_noise_cov

    #Generating a single sample with cov (vector of vector)
    sample = [[1.0, 2.0, 3.0]]
    obs_noise_cov = ones(3, 3)
    data_name = "d1"
    obs = Obs(sample, obs_noise_cov, data_name)
    @test obs.mean == sample[1]
    @test obs.obs_noise_cov == obs_noise_cov

    # Generate 1D-samples (data are columns of the row vector) -- this should result in scalar values for the mean and obs_noise_cov
    sample = reshape([1.0, 2.0, 3.0], 1, 3)
    data_name = "d1"
    obs = Obs(sample, data_name)
    @test obs.mean == 2.0
    @test obs.obs_noise_cov == 1.0

    # Generate 1D-samples (Vector of vector)
    sample = [[1.0], [2.0], [3.0]]
    data_name = "d1"
    obs = Obs(sample, data_name)
    @test obs.mean == 2.0
    @test obs.obs_noise_cov == 1.0

    # Generate 1D-samples, but also passing a known "1x1 covarince matrix" (stored as a variance)
    # (data are columns of the row vector)
    # this should result in scalar values for the mean and obs_noise_cov
    sample = reshape([1.0, 2.0, 3.0], 1, 3)
    obs_noise_cov = reshape([3.0], 1, 1) #needs to be of type Array{,2}
    data_name = "d1"
    obs = Obs(sample, obs_noise_cov, data_name)
    @test obs.mean == 2.0
    @test obs.obs_noise_cov == 3.0

    # Generate 1D-samples, but also passing a known "1x1 covarince matrix"
    # (data are vector of vectors)
    sample = [[1.0], [2.0], [3.0]]
    obs_noise_cov = reshape([3.0], 1, 1) #needs to be of type Array{,2}
    data_name = "d1"
    obs = Obs(sample, obs_noise_cov, data_name)
    @test obs.mean == 2.0
    @test obs.obs_noise_cov == 3.0

end
