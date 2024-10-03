using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
# using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses


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
    @test get_group_id(group) == Dict("1:8" => "1:20")

    u1c = [2, 4, 9, 10]
    g2 = 3:9
    group2 = UpdateGroup(u1c, g2)

    groups = [group1, group2]

    @test update_group_consistency(groups, input_dim, output_dim)

    # break consistency with different ways
    not_u1c = [2, 3, 4, 9, 10]
    group_bad = UpdateGroup([], g2)
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)
    group_bad = UpdateGroup(u1c, [])
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)
    group_bad = UpdateGroup(not_u1c, g2) # [u1,not_u1c] is not a partition of 1:10
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)
    not_inbd_g = [21] # outside of 1:20
    group_bad = UpdateGroup(u1c, not_inbd_g)
    @test_throws ArgumentError update_group_consistency([group1, group_bad], input_dim, output_dim)

end
