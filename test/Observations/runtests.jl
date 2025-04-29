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

@testset "covariance utilities" begin
    rng = Random.MersenneTwister(11023)
    testmat1 = randn(rng, 100, 100)
    svd_testmat1 = svd(testmat1)
    testmat2 = randn(rng, 5, 100)
    svd_testmat2 = svd(testmat2)
    r = 8
    mat_lr, mat_inv_lr = tsvd_mat(testmat1, r, return_inverse = true)
    @test all(isapprox.(mat_lr.S, svd_testmat1.S[1:r]))
    @test all(isapprox.(mat_inv_lr.S, 1.0 ./ svd_testmat1.S[1:r]))
    @test size(mat_lr.U) == (size(testmat1, 1), r)
    @test size(mat_lr.Vt) == (r, size(testmat1, 2))
    @test all(isapprox.(mat_inv_lr.U, mat_lr.V))
    @test all(isapprox.(mat_inv_lr.Vt, mat_lr.U'))

    badr = 106
    mat_lr, mat_inv_lr = tsvd_mat(testmat1, badr, return_inverse = true)
    @test all(isapprox.(mat_lr.S, svd_testmat1.S))
    @test all(isapprox.(mat_inv_lr.S, 1.0 ./ svd_testmat1.S))
    @test size(mat_lr.U) == size(testmat1)
    @test size(mat_lr.Vt) == size(testmat1)
    @test all(isapprox.(mat_inv_lr.U, mat_lr.V))
    @test all(isapprox.(mat_inv_lr.Vt, mat_lr.U'))

    mat_lr, mat_inv_lr = tsvd_mat(testmat2, return_inverse = true)
    @test all(isapprox.(mat_lr.S, svd_testmat2.S))
    @test all(isapprox.(mat_inv_lr.S, 1.0 ./ svd_testmat2.S))
    @test size(mat_lr.U) == (size(testmat2, 1), rank(testmat2))
    @test size(mat_lr.Vt) == (rank(testmat2), size(testmat2, 2))

    test_scale = 3.0 * I
    @test_throws ArgumentError tsvd_mat(test_scale)
    mat_lr, mat_inv_lr = tsvd_mat(test_scale, r, return_inverse = true)
    @test all(isapprox.(mat_lr.S, 3.0 * ones(r)))
    @test all(isapprox.(mat_inv_lr.S, 1.0 ./ 3.0 * ones(r)))

    # test getting the cov sqrt
    testmat3 = randn(rng, 100, 5)
    full_cov = cov(testmat3, dims = 2)
    pinv_full_cov = pinv(full_cov) # inv is not well defined, but psuedoinv is

    mat_lr, mat_inv_lr = tsvd_cov_from_samples(testmat3, return_inverse = true)
    @test norm(full_cov - mat_lr.U * Diagonal(mat_lr.S) * mat_lr.Vt) < 1e-10
    @test norm(pinv_full_cov - mat_inv_lr.U * Diagonal(mat_inv_lr.S) * mat_inv_lr.Vt) < 1e-10

    @test_throws ArgumentError tsvd_cov_from_samples(testmat3[:, 1:1])
    @test_logs (:warn,) tsvd_cov_from_samples(testmat3, data_are_columns = false)
    # create covariance from lowrank
    # full cov created by:
    samples = randn(rng, 100)

    obs_lr = Observation(Dict("samples" => samples, "covariances" => mat_lr, "names" => "lr"))

    obs_inv_lr = Observation(
        Dict("samples" => samples, "covariances" => mat_lr, "inv_covariances" => mat_inv_lr, "names" => "inv_lr"),
    )

    cov_lr = get_obs_noise_cov(obs_lr)
    cov_lr2 = get_obs_noise_cov(obs_inv_lr)
    @test norm(full_cov - cov_lr) < 1e-10
    @test norm(full_cov - cov_lr2) < 1e-10

    cov_inv_lr = get_obs_noise_cov_inv(obs_lr)
    cov_inv_lr2 = get_obs_noise_cov_inv(obs_inv_lr)
    @test norm(pinv_full_cov - cov_inv_lr) < 1e-10
    @test norm(pinv_full_cov - cov_inv_lr2) < 1e-10

    # even more reduced rank
    rk = 3
    mat_lr3 = tsvd_cov_from_samples(testmat3, rk)
    @test length(mat_lr3.S) == rk

    ## Sizes
    @test get_cov_size(full_cov) == 100
    @test get_cov_size(mat_lr) == 100

    ## Create SVDplusD
    @test_throws ArgumentError SVDplusD(mat_lr, 6.0 * Diagonal(ones(rk + 1))) # converts it to an SVD type

    dim = get_cov_size(mat_lr)
    X_I = SVDplusD(mat_lr, 6.0 * I) # converts it to an SVD type

    @test X_I == SVD(mat_lr.U, mat_lr.S .+ 6.0, mat_lr.Vt)
    a_diag = Diagonal(collect(1.0:dim))
    X_D = SVDplusD(mat_lr, a_diag)
    X_M = SVDplusD(mat_lr, Matrix(a_diag)) # converts it to a Diagonal type
    @test X_M == X_D
    @test size(X_D) == (dim, dim)
    @test get_cov_size(X_D) == dim
    @test get_svd_cov(X_D) == mat_lr
    @test get_diag_cov(X_D) == a_diag

    # Create DminusTall (the format for SVDplusD inverse)
    DmT = inv_cov(X_D) # stored in form D^-1 - R R^T where R is a tall matrix
    DmT2 = inv_cov(X_D) # stored in form D^-1 - R R^T where R is a tall matrix
    @test DmT == DmT2 #just check == implementation
    @test size(DmT) == (dim, dim)
    Dinv = inv(a_diag)
    S = Diagonal(mat_lr.S)
    U = mat_lr.U
    Vt = mat_lr.Vt
    T_nonsym = inv(S + S * Vt * Dinv * U * S) # often non-symmetric from rounding error
    cholT = cholesky(0.5 * (T_nonsym + T_nonsym'))
    tall = Dinv * U * S * cholT.L
    @test norm(get_diag_cov(DmT) - Dinv) < 1e-12
    @test norm(get_tall_cov(DmT) - tall) < 1e-12

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

    metadata = Dict("example1" => 4)


    # minibatcher 
    batch_size = 2
    rng = Random.MersenneTwister(11023)
    given_batches =
        [collect(((i - 1) * batch_size + 1):(i * batch_size)) for i in 1:Int(floor(maximum(sample_ids) / batch_size))]
    n_batches_per_epoch = maximum(sample_ids) / batch_size

    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    observation_series = ObservationSeries(obs_vec, minibatcher, series_names, metadata = metadata)
    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng))
    os_dict =
        Dict("observations" => obs_vec, "minibatcher" => minibatcher, "names" => series_names, "metadata" => metadata)
    bad_os_dict = Dict("minibatcher" => minibatcher, "names" => series_names, "metadata" => metadata)
    @test_throws ArgumentError ObservationSeries(bad_os_dict)
    observation_series_from_dict = ObservationSeries(os_dict)
    @test observation_series == observation_series_from_dict

    minibatcher = FixedMinibatcher(given_batches, "random", copy(rng)) #cant copy minibatcher yet
    new_epoch = create_new_epoch!(minibatcher, given_batches)
    @test get_observations(observation_series) == obs_vec
    @test get_minibatches(observation_series) == [new_epoch]
    @test get_current_minibatch_index(observation_series) == Dict("epoch" => 1, "minibatch" => 1)
    @test get_length_epoch(observation_series) == n_batches_per_epoch
    @test get_minibatch_index(observation_series, 1) == Dict("epoch" => 1, "minibatch" => 1)
    @test get_minibatch_index(observation_series, 8) ==
          Dict("epoch" => ((8 - 1) ÷ n_batches_per_epoch) + 1, "minibatch" => ((8 - 1) % n_batches_per_epoch) + 1)
    @test get_minibatcher(observation_series) == minibatcher
    @test get_names(observation_series) == series_names
    @test get_metadata(observation_series) == metadata

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
    @test isnothing(get_metadata(observation_series))

    # test the minibatch updating with epochs
    @test get_current_minibatch(observation_series) == new_epoch[1]
    @test get_minibatch(observation_series, 1) == new_epoch[1]
    for i in 2:5 # we have 5 batches
        update_minibatch!(observation_series)
        @test get_current_minibatch(observation_series) == new_epoch[i]
        @test get_minibatch(observation_series, i) == new_epoch[i]
    end
    update_minibatch!(observation_series)
    new_epoch2 = create_new_epoch!(minibatcher, given_batches)
    @test get_current_minibatch_index(observation_series) == Dict("epoch" => 2, "minibatch" => 1)
    @test get_current_minibatch(observation_series) == new_epoch2[1]
    @test get_minibatch(observation_series, 6) == new_epoch2[1]
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
    mb_idx = 6
    obs_minibatch = get_obs(observation_series, build = false)
    @test get_obs.(obs_vec[mb], build = false) == obs_minibatch
    obs_minibatch_stack = get_obs(observation_series)
    @test reduce(vcat, get_obs.(obs_vec[mb])) == obs_minibatch_stack
    @test get_obs(observation_series, mb_idx, build = false) == obs_minibatch
    @test get_obs(observation_series) == obs_minibatch_stack

    # get_obs_noise_cov
    obs_noise_cov_minibatch_blocks = get_obs_noise_cov(observation_series, build = false)
    minibatch_covs = []
    for observation in obs_vec[mb]
        push!(minibatch_covs, get_obs_noise_cov(observation, build = false))
    end
    minibatch_covs = reduce(vcat, minibatch_covs)
    @test minibatch_covs == obs_noise_cov_minibatch_blocks
    @test minibatch_covs == get_obs_noise_cov(observation_series, mb_idx, build = false)

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
    @test minibatch_cov_full == get_obs_noise_cov(observation_series, mb_idx)

    # just test the indexing a last time
    @test get_obs_noise_cov_inv(observation_series) == get_obs_noise_cov_inv(observation_series, mb_idx)


end

@testset "left-multiply utilities" begin
    # build some observations

    rng = Random.MersenneTwister(11023)

    samples = [randn(5), randn(7), randn(9), randn(12)]
    diag_cov = Diagonal(Float64.(collect(1:length(samples[1]))))
    Xrt = randn(length(samples[2]), length(samples[2]))
    full_mat_cov = Xrt * Xrt'

    testmat3 = randn(rng, length(samples[3]), 5)
    svd_cov = tsvd_cov_from_samples(testmat3)

    testmat4 = randn(rng, length(samples[4]), 6)
    svd_cov2 = tsvd_cov_from_samples(testmat4)
    dim = get_cov_size(svd_cov2)
    sum_cov = SVDplusD(svd_cov2, Diagonal(collect(1.0:dim)))

    covs = [diag_cov, full_mat_cov, svd_cov, sum_cov]
    names = ["d$(string(i))" for i in 1:length(samples)]
    obs_vec = []
    for (sam, cov, nam) in zip(samples, covs, names)
        push!(obs_vec, Observation(sam, cov, nam))
    end
    observation_series = ObservationSeries(obs_vec)
    observation_series_nosum = ObservationSeries(obs_vec[1:3])
    Γfull = zeros(33, 33)
    Γfull[1:5, 1:5] = diag_cov
    Γfull[6:12, 6:12] = full_mat_cov
    Γfull[13:21, 13:21] = svd_cov.Vt' * Diagonal(svd_cov.S) * svd_cov.Vt
    Γfull[22:33, 22:33] += svd_cov2.Vt' * Diagonal(svd_cov2.S) * svd_cov2.Vt
    Γfull[22:33, 22:33] += Diagonal(collect(1.0:12.0))
    Γ = get_obs_noise_cov(observation_series)
    @test norm(Γfull - Γ) < 1e-12

    Γfullinv = zeros(33, 33)
    Γfullinv[1:5, 1:5] = inv(diag_cov)
    Γfullinv[6:12, 6:12] = inv(full_mat_cov)
    Γfullinv[13:21, 13:21] = svd_cov.U * Diagonal(1.0 ./ svd_cov.S) * svd_cov.U'
    Γfullinv[22:33, 22:33] = inv(Γfull[22:33, 22:33]) #small enough to compute directly
    Γinv = get_obs_noise_cov_inv(observation_series)
    @test norm(Γfullinv - Γinv) < 1e-11
    # Test utilities for multiplying covariances without building them
    # test lmul_obs_noise_cov

    Xvec = randn(size(Γ, 1))
    X = randn(size(Γ, 1), 50)
    test1 = Γ * Xvec
    test2 = Γ * X
    @test norm(test1 - lmul_obs_noise_cov(observation_series, Xvec)) < 1e-11
    @test norm(test2 - lmul_obs_noise_cov(observation_series, X)) < 1e-11

    test1 = Γinv * Xvec
    test2 = Γinv * X
    @test norm(test1 - lmul_obs_noise_cov_inv(observation_series, Xvec)) < 1e-11
    @test norm(test2 - lmul_obs_noise_cov_inv(observation_series, X)) < 1e-11

    # test the pre-indexed versions:    
    # cov
    g_idx = [2, 3, 4, 8, 9, 15, 16, 24, 26, 27] # some indices from each cov-type
    test1 = Γ[g_idx, g_idx] * Xvec[g_idx]
    test2 = Γ[g_idx, g_idx] * X[g_idx, :]
    out1 = zeros(size(test1))
    out2 = zeros(size(test2))
    # The function is writte to be applied to pre-trimmed "X"s 
    lmul_obs_noise_cov!(out1, observation_series, Xvec[g_idx], g_idx)
    lmul_obs_noise_cov!(out2, observation_series, X[g_idx, :], g_idx)
    @test norm(test1 - out1) < 1e-11
    @test norm(test2 - out2) < 1e-11
    lmul_obs_noise_cov!(out1, observation_series, Xvec, g_idx)
    lmul_obs_noise_cov!(out2, observation_series, X, g_idx)
    @test norm(test1 - out1) < 1e-11
    @test norm(test2 - out2) < 1e-11


    # cov_inv
    test1 = Γinv[g_idx, g_idx] * Xvec[g_idx]
    test2 = Γinv[g_idx, g_idx] * X[g_idx, :]
    out1 = zeros(size(test1))
    out2 = zeros(size(test2))
    # In code this is applied to pre-trimmed "X"s or not
    lmul_obs_noise_cov_inv!(out1, observation_series, Xvec[g_idx], g_idx)
    lmul_obs_noise_cov_inv!(out2, observation_series, X[g_idx, :], g_idx)
    @test norm(test1 - out1) < 1e-11
    @test norm(test2 - out2) < 1e-11
    lmul_obs_noise_cov_inv!(out1, observation_series, Xvec, g_idx)
    lmul_obs_noise_cov_inv!(out2, observation_series, X, g_idx)
    @test norm(test1 - out1) < 1e-11
    @test norm(test2 - out2) < 1e-11

end
