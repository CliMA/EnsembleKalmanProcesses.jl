using Test
using Random
using Statistics
using LinearAlgebra
using EnsembleKalmanProcesses


@testset "Observations" begin

    # create observations
    sample_sizes = [3, 5, 20, 1]
    n_samples = length(sample_sizes)
    samples = []
    covariances = []
    inv_covariances = []
    for i in 1:n_samples
        push!(samples, vec(i * ones(sample_sizes[i])))
        if (i == 3)
            X = I
        elseif (i == 4)
            X = I
        else
            X = randn(sample_sizes[i], sample_sizes[i])
        end

        push!(covariances, i * X' * X)
        if !(i == 3) # take inverse if not == 3
            ic = inv(i * X' * X)
        else # here submit a user-defined inverse covariance (not true inverse)
            ic = I
        end
        push!(inv_covariances, ic)

    end
    names = ["d$(string(i))" for i in 1:n_samples]


    n_blocks = length(sample_sizes)
    indices = [1:sample_sizes[1]]
    if n_blocks > 1
        for i in 2:n_blocks
            push!(indices, (sum(sample_sizes[1:(i - 1)]) + 1):sum(sample_sizes[1:i]))
        end
    end

    # 1) via a dict [singleton] 
    obs_dict = Dict("samples" => samples[1], "covariances" => covariances[1], "names" => names[1])
    observation_1 = Observation(obs_dict)
    @test get_samples(observation_1) == [samples[1]] # all stored as a vec
    @test get_covs(observation_1) == [covariances[1]]
    @test all(isapprox.(get_inv_covs(observation_1)[1], inv_covariances[1], atol = 1e-10)) # inversion approximate
    @test get_names(observation_1) == [names[1]]
    @test get_indices(observation_1) == [indices[1]]

    # 2) via args [singleton] 
    observation_1 = Observation(samples[1], covariances[1], names[1])
    @test get_samples(observation_1) == [samples[1]] # all stored as a vec
    @test get_covs(observation_1) == [covariances[1]]
    @test all(isapprox.(get_inv_covs(observation_1)[1], inv_covariances[1], atol = 1e-10)) # inversion approximate
    @test get_names(observation_1) == [names[1]]
    @test get_indices(observation_1) == [indices[1]]

    # 2) via a dict [vec], pass in inv_covs
    obs_dict = Dict(
        "samples" => samples[2:4],
        "covariances" => covariances[2:4],
        "inv_covariances" => inv_covariances[2:4],
        "names" => names[2:4],
    )
    observation_2_4 = Observation(obs_dict)
    @test get_samples(observation_2_4) == samples[2:4]
    @test get_covs(observation_2_4) == covariances[2:4]
    @test get_inv_covs(observation_2_4) == inv_covariances[2:4]
    @test get_names(observation_2_4) == names[2:4]
    @test get_indices(observation_2_4) == [id .- maximum(indices[1]) for id in indices[2:4]] # shifted 

    # 3) via a list of args  (not pass inv_covs)
    observation_2_4_new = Observation(samples[2:4], covariances[2:4], names[2:4])
    @test get_samples(observation_2_4_new) == samples[2:4]
    @test get_covs(observation_2_4_new) == covariances[2:4]
    @test all(isapprox.(get_inv_covs(observation_2_4_new), inv.(covariances[2:4]), atol = 1e-10)) # inversion approximate
    @test get_names(observation_2_4_new) == names[2:4]
    @test get_indices(observation_2_4_new) == [id .- maximum(indices[1]) for id in indices[2:4]] # shifted 


    # 4) via combining Observations
    observation = combine_observations([observation_1, observation_2_4])
    @test get_samples(observation) == samples
    @test get_covs(observation) == covariances
    @test all(isapprox.(get_inv_covs(observation), inv_covariances, atol = 1e-10))
    @test get_names(observation) == names
    @test get_indices(observation) == indices # correctly shifted back

    # get_obs 
    obs_sample = get_obs(observation, build = false)
    obs_stacked = get_obs(observation) # default build=true
    @test obs_sample == get_samples(observation)
    @test obs_stacked == append!(get_samples(observation)...)

    # get_obs_noise_cov
    onc_block = get_obs_noise_cov(observation, build = false)
    onc_full = get_obs_noise_cov(observation) # default build=true
    @test onc_block == covariances

    full = zeros(maximum(indices[end]), maximum(indices[end]))
    for (idx, c) in zip(indices, covariances)
        if isa(c, UniformScaling)
            for idxx in idx
                full[idxx, idxx] = c.λ
            end
        else
            full[idx, idx] .= c
        end
    end
    @test onc_full == full

    # get_obs_noise_cov_inv
    onci_block = get_obs_noise_cov_inv(observation, build = false)
    onci_full = get_obs_noise_cov_inv(observation) # default build=true
    @test onci_block == inv_covariances

    full = zeros(maximum(indices[end]), maximum(indices[end]))
    for (idx, c) in zip(indices, inv_covariances)
        if isa(c, UniformScaling)
            for idxx in idx
                full[idxx, idxx] = c.λ
            end
        else
            full[idx, idx] .= c
        end
    end
    @test onci_full == full


end

@testset "Minibatching" begin

    m_size = 6
    n_samples = 20
    given_batches = [collect(((i - 1) * m_size + 1):(i * m_size)) for i in 1:n_samples]

    # 1) FixedMinibatcher
    fixed_minibatcher = FixedMinibatcher(given_batches[1])

    @test get_minibatches(fixed_minibatcher) == [given_batches[1]]
    @test get_method(fixed_minibatcher) == "order"
    @test get_rng(fixed_minibatcher) == Random.default_rng()
    new_epoch = create_new_epoch!(fixed_minibatcher)
    @test get_minibatches(fixed_minibatcher) == [given_batches[1]]
    @test new_epoch == [given_batches[1]]

    fixed_minibatcher = FixedMinibatcher(given_batches)

    @test get_minibatches(fixed_minibatcher) == given_batches
    @test get_method(fixed_minibatcher) == "order"
    @test get_rng(fixed_minibatcher) == Random.default_rng()
    new_epoch = create_new_epoch!(fixed_minibatcher)
    @test get_minibatches(fixed_minibatcher) == given_batches
    @test new_epoch == given_batches

    rng = Random.MersenneTwister(1102)
    method = "random"
    fixed_minibatcher = FixedMinibatcher(given_batches, method, copy(rng))
    @test get_minibatches(fixed_minibatcher) == given_batches
    @test get_method(fixed_minibatcher) == "random"
    @test get_rng(fixed_minibatcher) == rng
    new_epoch = create_new_epoch!(fixed_minibatcher)
    idx = shuffle(rng, collect(1:length(given_batches)))
    shuffled_batches = given_batches[idx]
    @test get_minibatches(fixed_minibatcher) == shuffled_batches
    @test new_epoch == shuffled_batches

    # 2) No minibatching - currently just make a Fixed batcher with 1 index
    default = no_minibatcher()
    @test get_minibatches(default) == [[1]]
    @test get_method(default) == "order"
    @test get_rng(default) == Random.default_rng()

    # 3) RandomFixedSizeMinibatcher

    minibatch_size = 7
    method = "trim"
    epoch = collect(1:100)
    rng = Random.MersenneTwister(12305)

    rfs_minibatcher = RandomFixedSizeMinibatcher(minibatch_size)
    @test get_minibatch_size(rfs_minibatcher) == minibatch_size
    @test get_method(rfs_minibatcher) == "extend"
    @test get_rng(rfs_minibatcher) == Random.default_rng()

    rfs_minibatcher = RandomFixedSizeMinibatcher(minibatch_size, method)
    @test get_minibatch_size(rfs_minibatcher) == minibatch_size
    @test get_method(rfs_minibatcher) == "trim"
    @test get_rng(rfs_minibatcher) == Random.default_rng()

    rfs_minibatcher = RandomFixedSizeMinibatcher(minibatch_size, method, copy(rng))
    @test get_minibatch_size(rfs_minibatcher) == minibatch_size
    @test get_method(rfs_minibatcher) == "trim"
    @test get_rng(rfs_minibatcher) == rng
    batched_epoch = create_new_epoch!(rfs_minibatcher, epoch)
    @test batched_epoch == get_minibatches(rfs_minibatcher)
    n_minibatches = Int(floor(length(epoch) / minibatch_size))
    indices = shuffle(copy(rng), epoch)
    trim_test = [indices[((i - 1) * minibatch_size + 1):(i * minibatch_size)] # bs sized minibatches
                 for i in 1:n_minibatches]
    @test batched_epoch == trim_test

    rfs_minibatcher = RandomFixedSizeMinibatcher(minibatch_size, copy(rng))
    batched_epoch = create_new_epoch!(rfs_minibatcher, epoch)
    @test batched_epoch == get_minibatches(rfs_minibatcher)
    n_minibatches = Int(floor(length(epoch) / minibatch_size))
    indices = shuffle(copy(rng), epoch)
    extend_test = [
        i < n_minibatches ? indices[((i - 1) * minibatch_size + 1):(i * minibatch_size)] : # bs sized minibatches
        indices[((n_minibatches - 1) * minibatch_size + 1):end] # final large minibatch < 2*bs sized
        for i in 1:n_minibatches
    ]
    @test batched_epoch == extend_test

end

@testset "ObservationSeries" begin

    # build some observations
    sample_sizes = [3, 5, 20, 1]
    n_samples = length(sample_sizes)

    sample_ids = 1:10
    obs_vec = []
    for sid in sample_ids
        samples = []
        covariances = []

        for i in 1:n_samples
            push!(samples, vec(sid .+ i * ones(sample_sizes[i])))

            X = randn(sample_sizes[i], sample_sizes[i])

            push!(covariances, sid .+ i * X' * X)
        end
        names = ["d$(string(i))" for i in 1:n_samples]

        push!(obs_vec, Observation(Dict("samples" => samples, "covariances" => covariances, "names" => names)))
    end
    series_names = ["obs_$(string(i))" for i in 1:n_samples]

    # minibatcher 
    batch_size = 2
    rng = Random.MersenneTwister(11023)
    given_batches =
        [collect(((i - 1) * batch_size + 1):(i * batch_size)) for i in 1:Int(floor(maximum(sample_ids) / batch_size))]
    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    observation_series = ObservationSeries(obs_vec, minibatcher, series_names)
    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    os_dict = Dict("observations" => obs_vec, "minibatcher" => minibatcher, "names" => series_names)
    observation_series_from_dict = ObservationSeries(os_dict)
    @test observation_series == observation_series_from_dict

    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng)) #cant copy minibatcher yet
    new_epoch = create_new_epoch!(minibatcher, given_batches)
    @test get_observations(observation_series) == obs_vec
    @test get_minibatches(observation_series) == [new_epoch]
    @test get_current_minibatch_index(observation_series) == Dict("epoch" => 1, "minibatch" => 1)
    @test get_minibatcher(observation_series) == minibatcher
    @test get_names(observation_series) == series_names

    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    observation_series = ObservationSeries(obs_vec, minibatcher)
    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    os_dict = Dict("observations" => obs_vec, "minibatcher" => minibatcher)
    observation_series_from_dict = ObservationSeries(os_dict)
    @test observation_series == observation_series_from_dict
    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))#cant copy minibatcher yet
    new_epoch = create_new_epoch!(minibatcher, given_batches)
    @test get_observations(observation_series) == obs_vec
    @test get_minibatches(observation_series) == [new_epoch]
    @test get_current_minibatch_index(observation_series) == Dict("epoch" => 1, "minibatch" => 1)
    @test get_minibatcher(observation_series) == minibatcher
    @test get_names(observation_series) == ["series_$(string(i))" for i in 1:length(obs_vec)]

    # test the minibatch updating with epochs
    @test get_current_minibatch(observation_series) == new_epoch[1]
    for i in 2:5 # we have 5 batches
        update_minibatch!(observation_series)
        @test get_current_minibatch(observation_series) == new_epoch[i]
    end
    update_minibatch!(observation_series)
    new_epoch2 = create_new_epoch!(minibatcher, given_batches)
    @test get_current_minibatch_index(observation_series) == Dict("epoch" => 2, "minibatch" => 1)
    @test get_current_minibatch(observation_series) == new_epoch2[1]
    @test get_minibatches(observation_series) == [new_epoch, new_epoch2]

    # test the no minibatch option
    observation_series_none = ObservationSeries(obs_vec)
    @test get_current_minibatch(observation_series_none) == collect(1:length(obs_vec))
    update_minibatch!(observation_series_none)
    @test get_current_minibatch(observation_series_none) == collect(1:length(obs_vec))
    @test get_current_minibatch_index(observation_series_none) == Dict("epoch" => 2, "minibatch" => 1)

    os_dict = Dict("observations" => obs_vec)
    observation_series_none_from_dict = ObservationSeries(os_dict) # no rng so can't compare to observation_series_none directly
    @test get_current_minibatch(observation_series_none_from_dict) == collect(1:length(obs_vec))
    update_minibatch!(observation_series_none_from_dict)
    @test get_current_minibatch(observation_series_none_from_dict) == collect(1:length(obs_vec))
    @test get_current_minibatch_index(observation_series_none_from_dict) == Dict("epoch" => 2, "minibatch" => 1)

    # get_obs (def: build = true)
    mb = new_epoch2[1]
    obs_minibatch = get_obs(observation_series, build = false)
    @test get_obs.(obs_vec[mb], build = false) == obs_minibatch
    obs_minibatch_stack = get_obs(observation_series)
    @test reduce(vcat, get_obs.(obs_vec[mb])) == obs_minibatch_stack

    # get_obs_noise_cov
    obs_noise_cov_minibatch_blocks = get_obs_noise_cov(observation_series, build = false)
    minibatch_covs = []
    for observation in obs_vec[mb]
        push!(minibatch_covs, get_obs_noise_cov(observation, build = false))
    end
    minibatch_covs = reduce(vcat, minibatch_covs)
    @test minibatch_covs == obs_noise_cov_minibatch_blocks

    obs_noise_cov_minibatch_full = get_obs_noise_cov(observation_series)
    minibatch_covs = []
    for observation in obs_vec[mb]
        push!(minibatch_covs, get_obs_noise_cov(observation))
    end
    block_sizes = size.(minibatch_covs, 1) # square mats
    minibatch_cov_full = zeros(sum(block_sizes), sum(block_sizes))
    idx_min = [0]
    for (i, mc) in enumerate(minibatch_covs)
        idx = (idx_min[1] + 1):(idx_min[1] + block_sizes[i])
        minibatch_cov_full[idx, idx] .= mc
        idx_min[1] += block_sizes[i]
    end
    @test minibatch_cov_full == obs_noise_cov_minibatch_full



end
