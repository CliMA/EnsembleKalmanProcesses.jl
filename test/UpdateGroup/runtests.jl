using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
# using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

@testset "UpdateGroups" begin

    input_dim = 10
    output_dim = 20
    u = 1:10
    g = 1:20

    # build a set of consistent update groups
    u1 = [1, 3, 5, 6, 7, 8]
    g1 = 1:20
    identifier = Dict("1:8" => "1:20")
    group1 = UpdateGroup(u1, g1, Dict("1:8" => "1:20"))
    @test get_u_group(group1) == u1
    @test get_g_group(group1) == g1
    @test get_group_id(group1) == Dict("1:8" => "1:20")

    u1c = [2, 4, 9, 10]
    g2 = 3:9
    group2 = UpdateGroup(u1c, g2)

    groups = [group1, group2]

    @test update_group_consistency(groups, input_dim, output_dim)

    # break consistency with different ways
    not_u1c = [2, 3, 4, 9, 10]
    @test_throws ArgumentError UpdateGroup([], g2)
    @test_throws ArgumentError group_bad = UpdateGroup(u1c, [])
    group_bad = UpdateGroup(not_u1c, g2) # [u1,not_u1c] is not a partition of 1:10
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)
    not_inbd_g = [21] # outside of 1:20
    group_bad = UpdateGroup(u1c, not_inbd_g)
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)

    # creation of update groups from prior and observations
    param_names = ["F", "G"]

    prior_F = ParameterDistribution(
        Dict(
            "name" => param_names[1],
            "distribution" => Parameterized(MvNormal([1.0, 0.0, -2.0], I)),
            "constraint" => repeat([bounded_below(0)], 3),
        ),
    ) # gives 3-D dist
    prior_G = constrained_gaussian(param_names[2], 5.0, 4.0, 0, Inf)
    priors = combine_distributions([prior_F, prior_G])

    # data 
    # given a list of vector statistics y and their covariances Γ 
    data_block_names = ["<X>", "<Y>"]

    observation_vec = []
    for i in 1:length(data_block_names)
        push!(
            observation_vec,
            Observation(Dict("samples" => vec(ones(i)), "covariances" => I(i), "names" => data_block_names[i])),
        )
    end
    observations = combine_observations(observation_vec)
    group_identifiers = Dict(["F"] => ["<X>"], ["G"] => ["<X>", "<Y>"])
    update_groups = create_update_groups(priors, observations, group_identifiers)


    param_names = get_name(priors)
    param_indices = batch(priors, function_parameter_opt = "dof")

    obs_names = get_names(observations)
    obs_indices = get_indices(observations)

    update_groups_test = []
    for (key, val) in group_identifiers

        key_vec = isa(key, AbstractString) ? [key] : key # make it iterable
        val_vec = isa(val, AbstractString) ? [val] : val

        u_group = []
        g_group = []
        for pn in key_vec
            pi = param_indices[pn .== param_names]
            push!(u_group, isa(pi, Int) ? [pi] : pi)
        end
        for obn in val_vec
            oi = obs_indices[obn .== obs_names]
            push!(g_group, isa(oi, Int) ? [oi] : oi)
        end
        u_group = reduce(vcat, reduce(vcat, u_group))
        g_group = reduce(vcat, reduce(vcat, g_group))
        push!(update_groups_test, UpdateGroup(u_group, g_group, Dict(key_vec => val_vec)))
    end

    @test update_groups == update_groups_test

    # throw errors
    bad_group_identifiers = Dict(["FF"] => ["<X>"], ["G"] => ["<X>", "<Y>"])
    @test_throws ArgumentError create_update_groups(priors, observations, bad_group_identifiers)
    bad_group_identifiers = Dict(["F"] => ["<XX>"], ["G"] => ["<X>", "<Y>"])
    @test_throws ArgumentError create_update_groups(priors, observations, bad_group_identifiers)
end
