using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random

using EnsembleKalmanProcesses.ParameterDistributions

@testset "ParameterDistributions" begin
    @testset "ParameterDistributionType" begin
        # Tests for the ParameterDistributionType
        d = Parameterized(Gamma(2.0, 0.8))
        @test d.distribution == Gamma(2.0, 0.8)

        d = Samples([1 2 3; 4 5 6])
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1 4; 2 5; 3 6]; params_are_columns = false)
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6])
        @test d.distribution_samples == [1 2 3 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6]; params_are_columns = false)
        @test d.distribution_samples == reshape([1, 2, 3, 4, 5, 6], :, 1)

    end
    @testset "ConstraintType" begin
        # Tests for the ConstraintType
        # The predefined transforms
        c1 = bounded_below(0.2)
        @test isapprox(c1.constrained_to_unconstrained(1.0) - (log(1.0 - 0.2)), 0)
        @test isapprox(c1.unconstrained_to_constrained(0.0) - (exp(0.0) + 0.2), 0)

        c2 = bounded_above(0.2)
        @test isapprox(c2.constrained_to_unconstrained(-1.0) - (log(0.2 - -1.0)), 0)
        @test isapprox(c2.unconstrained_to_constrained(10.0) - (0.2 - exp(10.0)), 0)


        c3 = bounded(-0.1, 0.2)
        @test isapprox(c3.constrained_to_unconstrained(0.0) - (log((0.0 - -0.1) / (0.2 - 0.0))), 0)
        @test isapprox(c3.unconstrained_to_constrained(1.0) - ((0.2 * exp(1.0) + -0.1) / (exp(1.0) + 1)), 0)
        @test_throws DomainError bounded(0.2, -0.1)

        #an example with user defined invertible transforms
        c_to_u = (x -> 3 * x + 14)
        u_to_c = (x -> (x - 14) / 3)

        c4 = Constraint(c_to_u, u_to_c)
        @test isapprox(c4.constrained_to_unconstrained(5.0) - c_to_u(5.0), 0)
        @test isapprox(c4.unconstrained_to_constrained(5.0) - u_to_c(5.0), 0)

        #length, size
        @test length(c1) == 1
        @test size(c1) == (1,)

    end

    @testset "ParameterDistribution: Build and combine" begin

        # create single ParameterDistribution
        d = Parameterized(Gamma(2.0, 0.8))
        c_mismatch = [no_constraint(), no_constraint()]
        c_wrongtype = [3.0]
        c = no_constraint()
        name = "unconstrained_Gamma"

        @test_throws ArgumentError ParameterDistribution(d, c_wrongtype, name) #wrong type of constraint
        @test_throws DimensionMismatch ParameterDistribution(d, c_mismatch, name) #wrong number of constraints

        # test checks on stored information
        u = ParameterDistribution(d, c, name)
        @test u.distribution == [d]
        @test u.constraint == [c]
        @test u.name == [name]

        d = Parameterized(Normal(0, 1))
        c = [no_constraint()]
        u = ParameterDistribution(d, c, name)
        @test u.constraint == c #as c is already a vector

        # test concatenation
        d1 = Parameterized(Normal(0, 1))
        c1 = no_constraint()
        name1 = "unconstrained_normal"
        u1 = ParameterDistribution(d1, c1, name1)
        @test u1.constraint == [c1]

        d2 = Parameterized(MvNormal(3, 0.2))
        c2 = repeat([no_constraint()], 3) #3D distribution
        name2 = "3d_unconstrained_MvNs"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c3 = [bounded(10, 15), no_constraint()]
        name3 = "constrained_sampled"
        u3 = ParameterDistribution(d3, c3, name3)

        @test_throws ArgumentError ParameterDistribution([u1, u2])

        u = combine_distributions([u1, u2, u3])
        @test u.distribution == [d1, d2, d3]
        @test u.constraint == cat([[c1], c2, c3]..., dims = 1)
        @test u.name == [name1, name2, name3]

    end

    @testset "ParameterDistribution: Dictionary interface" begin

        # dictionary - testing test checks on the distribution elements
        d = Parameterized(Normal(0, 1))
        c = no_constraint()
        name = "unconstrained_normal"

        param_dict_fail = [Dict("a" => 1), 1]
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not an array of Dicts

        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not a Dict

        param_dict_fail = Dict("distribution" => d)
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not all the keys
        param_dict_fail = Dict("distribution" => 1, "constraint" => c, "name" => name)
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not right distribution type
        param_dict_fail = Dict("distribution" => d, "constraint" => 1, "name" => name)
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not right constraint type
        param_dict_fail = Dict("distribution" => d, "constraint" => [1, no_constraint(), 3], "name" => name)
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not right constraint type
        param_dict_fail = Dict("distribution" => d, "constraint" => c, "name" => 1)
        @test_throws ArgumentError ParameterDistribution(param_dict_fail) # not right name type
        param_dict_fail = Dict("distribution" => d, "constraint" => [no_constraint(), no_constraint()], "name" => name)
        @test_throws DimensionMismatch ParameterDistribution(param_dict_fail) # wrong number of constraints

        param_dict = Dict("distribution" => d, "constraint" => c, "name" => name)
        u = ParameterDistribution(param_dict)
        @test u.distribution == [d]
        @test u.constraint == [c]
        @test u.name == [name]

        c_arr = [no_constraint()]
        param_dict = Dict("distribution" => d, "constraint" => c_arr, "name" => name)
        u = ParameterDistribution(param_dict)
        @test u.constraint == [c]

        # Tests for the ParameterDistribution
        d = Parameterized(MvNormal(4, 0.1))
        c = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name = "constrained_mvnormal"
        param_dict = Dict("distribution" => d, "constraint" => c, "name" => name)

        u = ParameterDistribution(param_dict)
        @test u.distribution == [d]
        @test u.constraint == c
        @test u.name == [name]
        param_dict_fail = param_dict
        param_dict_fail["constraint"] = c[1]
        @test_throws DimensionMismatch ParameterDistribution(param_dict_fail) # wrong number of constraints

        # Tests for the ParameterDistribution
        d1 = Parameterized(MvNormal(4, 0.1))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        param_dict1 = Dict("distribution" => d1, "constraint" => c1, "name" => name1)

        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [bounded(10, 15), no_constraint()]
        name2 = "constrained_sampled"
        param_dict2 = Dict("distribution" => d2, "constraint" => c2, "name" => name2)

        param_dict_array = [param_dict1, param_dict2]
        u = ParameterDistribution(param_dict_array)
        @test u.distribution == [d1, d2]
        @test u.constraint == cat([c1, c2]..., dims = 1)
        @test u.name == [name1, name2]

    end

    @testset "getter functions" begin
        # setup for the tests:
        d1 = Parameterized(MvNormal(4, 0.1))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        u = combine_distributions([u1, u2])

        # Test for get_dimension(s)
        @test ndims(u1) == 4
        @test ndims(u2) == 1
        @test ndims(u) == 5
        @test get_dimensions(u1) == [4]
        @test get_dimensions(u2) == [1]
        @test get_dimensions(u) == [4, 1]

        # Tests for get_name
        @test get_name(u1) == [name1]
        @test get_name(u) == [name1, name2]

        # Tests for get_n_samples
        @test typeof(get_n_samples(u)[name1]) <: String
        @test get_n_samples(u)[name2] == 4

        # Tests for get_distribution
        @test get_distribution(d1) == MvNormal(4, 0.1)
        @test get_distribution(u1)[name1] == MvNormal(4, 0.1)
        @test typeof(get_distribution(d2)) == Array{Int64, 2}
        @test typeof(get_distribution(u2)[name2]) == Array{Int64, 2}

        d = get_distribution(u)
        @test d[name1] == MvNormal(4, 0.1)
        @test typeof(d[name2]) == Array{Int64, 2}

        # Test for get_all_constraints
        @test get_all_constraints(u) == cat([c1, c2]..., dims = 1)
    end

    @testset "statistics functions" begin

        # setup for the tests:
        d1 = Parameterized(MvNormal(4, 0.1))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"

        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = Parameterized(Beta(2, 2))
        c3 = no_constraint()
        name3 = "unconstrained_beta"
        u3 = ParameterDistribution(d3, c3, name3)

        u = combine_distributions([u1, u2])

        d4 = Samples([1 2 3 4 5 6 7 8; 8 7 6 5 4 3 2 1])
        c4 = [no_constraint(), no_constraint()]
        name4 = "constrained_MVsampled"
        u4 = ParameterDistribution(d4, c4, name4)

        v = combine_distributions([u1, u2, u3, u4])

        # Tests for sample distribution
        seed = 2020
        Random.seed!(seed)
        s1 = rand(MvNormal(4, 0.1), 1)
        Random.seed!(seed)
        @test sample(u1) == s1

        Random.seed!(seed)
        s1 = rand(MvNormal(4, 0.1), 3)
        Random.seed!(seed)
        @test sample(u1, 3) == s1

        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 1)
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample(u2) == s2

        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 3)
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample(u2, 3) == s2

        Random.seed!(seed)
        s1 = sample(u1, 3)
        s2 = sample(u2, 3)
        s3 = sample(u3, 3)
        s4 = sample(u4, 3)
        Random.seed!(seed)
        s = sample(v, 3)
        @test s == cat([s1, s2, s3, s4]..., dims = 1)

        #Test for get_logpdf
        @test_throws ErrorException get_logpdf(u, zeros(ndims(u)))
        x_in_bd = [0.5]
        Random.seed!(seed)
        lpdf3 = logpdf.(Beta(2, 2), x_in_bd)[1] #throws deprecated warning without "."

        Random.seed!(seed)
        @test isapprox(get_logpdf(u3, x_in_bd) - lpdf3, 0.0; atol = 1e-6)
        @test_throws DimensionMismatch get_logpdf(u3, [0.5, 0.5])

        #Test for cov, var        
        block_cov = cat([cov(d1), var(d2), var(d3), cov(d4)]..., dims = (1, 2))
        @test isapprox(cov(v) - block_cov, zeros(ndims(v), ndims(v)); atol = 1e-6)
        block_var = [block_cov[i, i] for i in 1:size(block_cov)[1]]
        @test isapprox(var(v) - block_var, zeros(ndims(v)); atol = 1e-6)

        #Test for mean
        means = cat([mean(d1), mean(d2), mean(d3), mean(d4)]..., dims = 1)
        @test isapprox(mean(v) - means, zeros(ndims(v)); atol = 1e-6)

    end

    @testset "statistics functions: explict RNG" begin

        # setup for the tests:
        rng_seed = 1234
        test_d = MvNormal(4, 0.1)
        test_d3a = Beta(2, 2)
        test_d3b = MvNormal(2, 0.1)
        d0 = Parameterized(test_d)

        d1 = Parameterized(test_d)
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        # Tests for sample distribution
        rng1 = Random.MersenneTwister(rng_seed)
        @test sample(copy(rng1), d0) == rand(copy(rng1), test_d, 1)
        @test sample(copy(rng1), d0, 3) == rand(copy(rng1), test_d, 3)

        @test sample(copy(rng1), u1) == rand(copy(rng1), test_d, 1)
        @test sample(copy(rng1), u1, 3) == rand(copy(rng1), test_d, 3)

        idx = StatsBase.sample(copy(rng1), collect(1:size(d2.distribution_samples)[2]), 1)
        s2 = d2.distribution_samples[:, idx]
        @test sample(copy(rng1), u2) == s2
        @test sample(copy(rng1), u2, 1) == s2
        @test sample(copy(rng1), d2) == s2
        @test sample(copy(rng1), d2, 1) == s2

        # try it again with different RNG; use StableRNG since Random doesn't provide a 
        # second seedable algorithm on julia <=1.7
        rng2 = StableRNG(rng_seed)
        @test sample(copy(rng2), d0) == rand(copy(rng2), test_d, 1)
        @test sample(copy(rng2), d0, 3) == rand(copy(rng2), test_d, 3)

        @test sample(copy(rng2), u1) == rand(copy(rng2), test_d, 1)
        @test sample(copy(rng2), u1, 3) == rand(copy(rng2), test_d, 3)

        idx = StatsBase.sample(copy(rng2), collect(1:size(d2.distribution_samples)[2]), 1)
        s2 = d2.distribution_samples[:, idx]
        @test sample(copy(rng2), u2) == s2

        # test that optional parameter defaults to Random.GLOBAL_RNG, for all methods.
        # reset the global seed instead of copying the rng object's state
        rng_seed = 2468
        Random.seed!(rng_seed)
        test_lhs = sample(d0)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        test_lhs = sample(d0, 3)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 3)

        Random.seed!(rng_seed)
        test_lhs = sample(u1)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        test_lhs = sample(u1, 3)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 3)

        Random.seed!(rng_seed)
        test_lhs = sample(d0)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 1)
        test_lhs = d2.distribution_samples[:, idx]
        Random.seed!(rng_seed)
        @test test_lhs == sample(u2)
    end

    @testset "transform functions" begin
        #setup for the tests
        tol = 1e-8
        d1 = Parameterized(MvNormal(4, 0.1))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        param_dict1 = Dict("distribution" => d1, "constraint" => c1, "name" => name1)
        u1 = ParameterDistribution(param_dict1)

        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [bounded(10, 15), no_constraint()]
        name2 = "constrained_sampled"
        param_dict2 = Dict("distribution" => d2, "constraint" => c2, "name" => name2)
        u2 = ParameterDistribution(param_dict2)

        param_dict = [param_dict1, param_dict2]
        u = ParameterDistribution(param_dict)

        x_unbd = rand(MvNormal(6, 3), 1000)  #6 x 1000 
        # Tests for transforms
        x_real_constrained1 = mapslices(x -> transform_unconstrained_to_constrained(u1, x), x_unbd[1:4, :]; dims = 1)
        @test isapprox(
            x_unbd[1:4, :] -
            mapslices(x -> transform_constrained_to_unconstrained(u1, x), x_real_constrained1; dims = 1),
            zeros(size(x_unbd[1:4, :]));
            atol = tol,
        )

        x_real_constrained2 = mapslices(x -> transform_unconstrained_to_constrained(u2, x), x_unbd[5:6, :]; dims = 1)
        @test isapprox(
            x_unbd[5:6, :] -
            mapslices(x -> transform_constrained_to_unconstrained(u2, x), x_real_constrained2; dims = 1),
            zeros(size(x_unbd[5:6, :]));
            atol = tol,
        )

        x_real = mapslices(x -> transform_unconstrained_to_constrained(u, x), x_unbd; dims = 1)
        x_unbd_tmp = mapslices(x -> transform_constrained_to_unconstrained(u, x), x_real; dims = 1)
        @test isapprox(x_unbd - x_unbd_tmp, zeros(size(x_unbd)); atol = tol)

        # Tests transforms for other input structures
        @test isapprox(transform_unconstrained_to_constrained(u1, x_unbd[1:4, :]), x_real_constrained1; atol = tol)
        @test isapprox(
            x_unbd[1:4, :] - transform_constrained_to_unconstrained(u1, x_real_constrained1),
            zeros(size(x_unbd[1:4, :]));
            atol = tol,
        )
        @test isapprox(transform_unconstrained_to_constrained(u2, x_unbd[5:6, :]), x_real_constrained2; atol = tol)
        @test isapprox(
            x_unbd[5:6, :] - transform_constrained_to_unconstrained(u2, x_real_constrained2),
            zeros(size(x_unbd[5:6, :]));
            atol = tol,
        )
        @test isapprox(
            transform_unconstrained_to_constrained(u2, [x_unbd[5:6, :], x_unbd[5:6, :]]),
            [x_real_constrained2, x_real_constrained2];
            atol = tol,
        )
    end
end
