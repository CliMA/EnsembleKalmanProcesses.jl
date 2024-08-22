using Test
using Distributions
using StatsBase
import GaussianRandomFields # we wrap this so we don't want to use "using"
const GRF = GaussianRandomFields
using StableRNGs
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

        # Vector container - 
        d = VectorOfParameterized(repeat([Normal(0, 1)], 5))
        @test d.distribution == repeat([Normal(0, 1)], 5)
    end

    @testset "ParameterDistribution: over function spaces" begin

        # create a 1D Matern GRF with GaussianRandomFields.jl 
        # For now we only support 1D fields
        dim = 1
        smoothness = 1.0
        corr_length = 0.1
        dofs = 30
        pts = collect(0:0.02:1)
        grf = GRF.GaussianRandomField(
            GRF.CovarianceFunction(dim, GRF.Matern(corr_length, smoothness)),
            GRF.KarhunenLoeve(dofs),
            pts,
        )

        #our wrapper for learning the field:
        pkg = GRFJL()
        d = GaussianRandomFieldInterface(grf, pkg)

        coeff_prior = constrained_gaussian("GRF_coefficients", 0.0, 1.0, -Inf, Inf, repeats = dofs)
        @test get_grf(d) == grf
        @test get_distribution(d) == coeff_prior
        @test get_package(d) == pkg

        # and with another prior:
        wide_pos_prior = constrained_gaussian("GRF_coefficients_wide_pos", 0.0, 10.0, -10, Inf, repeats = dofs)
        d_wide_pos = GaussianRandomFieldInterface(grf, pkg, wide_pos_prior)
        @test get_distribution(d_wide_pos) == wide_pos_prior
        err_prior = constrained_gaussian("GRF_coefficients_wide_pos", 0.0, 10.0, -10, Inf, repeats = dofs + 1) # wrong num dofs
        @test_throws ArgumentError GaussianRandomFieldInterface(grf, pkg, err_prior)


        #function-based utils
        @test spectrum(d) == grf.data #evalues and efunctions
        @test eval_pts(d) == grf.pts
        @test input_dims(d) == dim
        @test n_dofs(d) == dofs
        @test n_eval_pts(d) == prod([length(grf.pts[i]) for i in 1:dim])

        #coeff-based utils
        @test mean(d) == mean(coeff_prior)
        @test cov(d) == cov(coeff_prior)
        @test var(d) == var(coeff_prior)
        @test logpdf(d, ones(dofs)) ≈ logpdf(coeff_prior, ones(dofs))
        @test get_n_samples(d) == get_n_samples(coeff_prior) # if coeff was a Samples dist


        # ndims now has 3 options
        @test ndims(d) == n_dofs(d)
        @test ndims(d, function_parameter_opt = "constraint") == 1 #to resolve dimension differences between coeff and function spaces 
        @test ndims(d, function_parameter_opt = "eval") == n_eval_pts(d) #to resolve dimension differences between coeff and function spaces
        @test_throws ArgumentError ndims(d, function_parameter_opt = "other")

        #sampling 
        s = 9211
        n_sample = 6

        # sample dist
        Random.seed!(s)
        sam = sample(coeff_prior)
        Random.seed!(s)
        @test sample(d) ≈ sam
        Random.seed!(s)
        sam2 = sample(coeff_prior, n_sample)
        Random.seed!(s)
        @test sample(d, n_sample) ≈ sam2

        rng1 = Random.MersenneTwister(s)
        sam = sample(copy(rng1), coeff_prior)
        sam2 = sample(copy(rng1), coeff_prior, n_sample)
        @test sample(copy(rng1), d) ≈ sam
        @test sample(copy(rng1), d, n_sample) ≈ sam2

        # test building functions from parameters
        n_pts = n_eval_pts(d)
        tol = 1e8 * eps() # for the wide priors, the exponential transforms can produces less accurate recovery.

        # [a] build function samples (Random.GLOBAL_RNG) 
        Random.seed!(s)
        xi1 = sample(coeff_prior)
        sample1 = reshape(GRF.sample(grf, xi = xi1), :, 1) # deterministic with xi input

        Random.seed!(s)
        @test build_function_sample(d) ≈ sample1 atol = tol # build with prior and default RNG
        @test build_function_sample(d, xi1) ≈ sample1 atol = tol # pass coeffs explicitly
        Random.seed!(s)
        xi2 = sample(coeff_prior, n_sample)
        sample2 = zeros(n_pts, n_sample)
        for i in 1:n_sample
            sample2[:, i] = GRF.sample(grf, xi = xi2[:, i])[:]
        end
        Random.seed!(s)
        @test build_function_sample(d, n_sample) ≈ sample2 atol = tol # build with prior and default RNG
        @test build_function_sample(d, xi2, n_sample) ≈ sample2 atol = tol # pass coeffs explicitly


        # [b] building function samples from the prior (with explicit rng)
        rng1 = Random.MersenneTwister(s)
        rng2 = copy(rng1)
        coeff_mat = sample(rng2, wide_pos_prior, 1)
        constrained_coeff_mat = transform_unconstrained_to_constrained(wide_pos_prior, coeff_mat)
        sample5 = reshape(GRF.sample(grf, xi = constrained_coeff_mat), :, 1)

        rng3 = copy(rng1)
        coeff_mat2 = sample(rng3, wide_pos_prior, n_sample)
        constrained_coeff_mat2 = transform_unconstrained_to_constrained(wide_pos_prior, coeff_mat2)
        sample6 = zeros(n_pts, n_sample)
        for i in 1:n_sample
            sample6[:, i] = GRF.sample(grf, xi = constrained_coeff_mat2[:, i])[:]
        end
        @test build_function_sample(copy(rng1), d_wide_pos) ≈ sample5 atol = tol
        @test build_function_sample(d_wide_pos, constrained_coeff_mat) ≈ sample5 atol = tol

        @test all(isapprox.(build_function_sample(copy(rng1), d_wide_pos, n_sample), sample6, atol = tol))
        @test all(isapprox.(build_function_sample(d_wide_pos, constrained_coeff_mat2, n_sample), sample6, atol = tol))

        @test_throws DimensionMismatch build_function_sample(d_wide_pos, constrained_coeff_mat2, n_sample + 1)

        if TEST_PLOT_OUTPUT
            # plot the samples
            plt = plot(pts, build_function_sample(copy(rng1), d, 20), legend = false) #uses the coeff_prior
            savefig(plt, joinpath(@__DIR__, "GRF_samples.png"))
        end

        # Put within parameter distribution
        function_constraint = bounded_below(3)
        pd = ParameterDistribution(
            Dict("distribution" => d_wide_pos, "name" => "grf_above_3", "constraint" => function_constraint),
        )

        # "test" show
        show(pd)

        # Transforms:
        # u->c goes from coefficients to constrained function evaluations. by default
        sample5_constrained = function_constraint.unconstrained_to_constrained.(sample5)
        sample6_constrained = function_constraint.unconstrained_to_constrained.(sample6)
        sample5_constrained_direct = transform_unconstrained_to_constrained(pd, vec(coeff_mat))
        sample6_constrained_direct = transform_unconstrained_to_constrained(pd, coeff_mat2)
        @test sample5_constrained ≈ sample5_constrained_direct atol = tol
        @test sample6_constrained ≈ sample6_constrained_direct atol = tol

        # specifying from unc. to cons. function evaluations with flag
        @test sample5_constrained ≈ transform_unconstrained_to_constrained(pd, vec(sample5), build_flag = false)
        @test all(
            isapprox.(
                sample6_constrained,
                transform_unconstrained_to_constrained(pd, sample6, build_flag = false),
                atol = tol,
            ),
        )

        # c->u is the inverse, of the build_flag=false u->c ONLY
        @test sample5 ≈ transform_constrained_to_unconstrained(pd, vec(sample5_constrained)) atol = tol
        biggertol = 1e-5
        @test all(isapprox.(sample6, transform_constrained_to_unconstrained(pd, sample6_constrained), atol = biggertol)) #can be sensitive to sampling (sometimes throws a near "Inf" so inverse is less accurate)



        if TEST_PLOT_OUTPUT
            dim_plot = 2
            dofs_plot = 30
            pts_plot = [collect(0:0.01:1), collect(1:0.02:2)]

            grf_plot = GRF.GaussianRandomField(
                GRF.CovarianceFunction(dim_plot, GRF.Matern(0.05, 2)),
                GRF.KarhunenLoeve(dofs_plot),
                pts_plot...,
            )
            d_plot = GaussianRandomFieldInterface(grf_plot, pkg)
            pd_plot = ParameterDistribution(
                Dict("distribution" => d_plot, "name" => "pd_min5_5", "constraint" => bounded(-5, -3)),
            )
            sample_constrained_flat = transform_unconstrained_to_constrained(pd_plot, ones(dofs_plot))
            sample_unconstrained_flat = transform_constrained_to_unconstrained(pd_plot, sample_constrained_flat)
            shape = [length(pp) for pp in pts_plot]
            # plot the 2D samples. remember heatmap requires a transpose...
            plt3 = contour(pts_plot..., reshape(sample_unconstrained_flat, shape...)', fill = true)
            savefig(plt3, joinpath(@__DIR__, "GRF_samples_unconstrained.png"))
            plt4 = contour(pts_plot..., reshape(sample_constrained_flat, shape...)', fill = true)
            savefig(plt4, joinpath(@__DIR__, "GRF_samples_constrained.png"))

        end


    end


    @testset "ConstraintType" begin
        tol = 10 * eps(Float64)
        # Tests for the ConstraintType
        # The predefined transforms
        c1 = bounded_below(0.2)
        @test isapprox(c1.constrained_to_unconstrained(1.0) - (log(1.0 - 0.2)), 0.0, atol = tol)
        @test isapprox(c1.unconstrained_to_constrained(0.0) - (exp(0.0) + 0.2), 0.0, atol = tol)
        @test get_constraint_type(c1) == BoundedBelow
        @test get_bounds(c1) == Dict("lower_bound" => 0.2)

        c2 = bounded_above(0.2)
        @test isapprox(c2.constrained_to_unconstrained(-1.0) - (-log(0.2 - -1.0)), 0.0, atol = tol)
        @test isapprox(c2.unconstrained_to_constrained(10.0) - (0.2 - exp(-10.0)), 0.0, atol = tol)
        @test get_constraint_type(c2) == BoundedAbove
        @test get_bounds(c2) == Dict("upper_bound" => 0.2)

        c3 = bounded(-0.1, 0.2)
        @test get_bounds(c3) == Dict("lower_bound" => -0.1, "upper_bound" => 0.2)
        @test get_constraint_type(c3) == Bounded
        @test isapprox(c3.constrained_to_unconstrained(0.0) - (log((0.0 - -0.1) / (0.2 - 0.0))), 0.0, atol = tol)
        @test isapprox(
            c3.unconstrained_to_constrained(1.0) - ((0.2 * exp(1.0) + -0.1) / (exp(1.0) + 1)),
            0.0,
            atol = tol,
        )
        @test_throws DomainError bounded(0.2, -0.1)

        #an example with user defined invertible transforms
        c_to_u = (x -> 3 * x + 14)
        jacobian = (x -> 3)
        u_to_c = (x -> (x - 14) / 3)

        abstract type MyConstraint <: ConstraintType end
        c4 = Constraint{MyConstraint}(c_to_u, jacobian, u_to_c, nothing)
        @test isapprox(c4.constrained_to_unconstrained(5.0) - c_to_u(5.0), 0.0, atol = tol)
        @test isapprox(c4.unconstrained_to_constrained(5.0) - u_to_c(5.0), 0.0, atol = tol)
        @test get_constraint_type(c4) == MyConstraint

        #length, size
        @test length(c1) == 1
        @test size(c1) == (1,)

        #equality
        @test c3 == c3
        @test !(c1 == c2)

    end

    @testset "ParameterDistribution: Build and combine" begin

        # create single ParameterDistribution
        d = VectorOfParameterized([Normal(0, 1), Gamma(2.0, 0.8)])
        c_mismatch = no_constraint()
        c_wrongtype = [no_constraint(), 3.0]
        c = [no_constraint(), no_constraint()]
        name = "normal_and_gamma"

        @test_throws ArgumentError ParameterDistribution(d, c_wrongtype, name) #wrong type of constraint
        @test_throws DimensionMismatch ParameterDistribution(d, c_mismatch, name) #wrong number of constraints

        # test checks on stored information
        u = ParameterDistribution(d, c, name)
        @test u.distribution == [d]
        @test u.constraint == c #as c is already a vector
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

        d2 = VectorOfParameterized(repeat([MvNormal(ones(3), 0.2 * I)], 4))
        c2 = repeat([no_constraint()], 12) #3D distribution repeated 4 times has 12 constraints
        name2 = "three_unconstrained_MvNs"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c3 = [bounded(10, 15), no_constraint()]
        name3 = "constrained_sampled"
        u3 = ParameterDistribution(d3, c3, name3)

        d4 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c4 = [bounded(10, 15), no_constraint()]
        name4 = "constrained_sampled"
        u4 = ParameterDistribution(d4, c4, name4)

        dim = 1
        dofs = 10
        pts = collect(0:0.01:1)
        pkg = GRFJL()
        grf = GRF.GaussianRandomField(GRF.CovarianceFunction(dim, GRF.Matern(0.05, 1)), GRF.KarhunenLoeve(dofs), pts)
        d5 = GaussianRandomFieldInterface(grf, pkg)
        c5 = bounded(-5, 5)
        name5 = "grf_in_min5_5"
        u5 = ParameterDistribution(d5, c5, name5)



        @test_throws ArgumentError ParameterDistribution([u1, u2])

        u = combine_distributions([u1, u2, u3, u4, u5])
        @test u.distribution == [d1, d2, d3, d4, d5]
        @test u.constraint == cat([[c1], c2, c3, c4, c5]..., dims = 1)
        @test u.name == [name1, name2, name3, name4, name5]

        # "test" show
        show(u)

        #equality
        @test u == u
        @test !(u2 == u3)
        @test u3 == u4


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
        d = Parameterized(MvNormal(zeros(4), 0.1 * I))
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
        d1 = Parameterized(MvNormal(zeros(4), 0.1 * I))
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
        d1 = Parameterized(MvNormal(zeros(4), 0.1 * I))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = VectorOfParameterized(repeat([Beta(2, 2)], 3))
        c3 = repeat([no_constraint()], 3)
        name3 = "vector_beta"
        u3 = ParameterDistribution(d3, c3, name3)

        u = combine_distributions([u1, u2, u3])

        # Test for get_dimension(s)
        @test ndims(u1) == 4
        @test ndims(u2) == 1
        @test ndims(u) == 8

        @test get_dimensions(u1) == [4]
        @test get_dimensions(u2) == [1]
        @test get_dimensions(u) == [4, 1, 3]

        # Tests for get_name
        @test get_name(u1) == [name1]
        @test get_name(u) == [name1, name2, name3]

        # Tests for get_n_samples
        @test typeof(get_n_samples(u)[name1]) <: String
        @test typeof(get_n_samples(u)[name3]) <: String
        @test get_n_samples(u)[name2] == 4

        # Tests for get_distribution
        @test get_distribution(d1) == MvNormal(zeros(4), 0.1 * I)
        @test get_distribution(u1)[name1] == MvNormal(zeros(4), 0.1 * I)
        @test typeof(get_distribution(d2)) == Array{Int64, 2}
        @test typeof(get_distribution(u2)[name2]) == Array{Int64, 2}

        @test get_distribution(d3) == repeat([Beta(2, 2)], 3)
        @test get_distribution(u3)[name3] == repeat([Beta(2, 2)], 3)

        d = get_distribution(u)
        @test d[name1] == MvNormal(zeros(4), 0.1 * I)
        @test typeof(d[name2]) == Array{Int64, 2}

        # Test for get_all_constraints
        @test get_all_constraints(u) == cat([c1, c2, c3]..., dims = 1)
        constraint_dict = Dict(name1 => c1, name2 => c2, name3 => c3)
        @test get_all_constraints(u; return_dict = true) == constraint_dict
    end

    @testset "statistics functions" begin

        # setup for the tests:
        d1 = Parameterized(MvNormal(zeros(4), 0.1 * I))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = VectorOfParameterized([Beta(2, 2), MvNormal(zeros(2), 0.1 * I)])
        c3 = [no_constraint(), no_constraint(), no_constraint()] # d3 has 3 dimensions
        name3 = "unconstrained_beta_and_MvN"
        u3 = ParameterDistribution(d3, c3, name3)

        u = combine_distributions([u1, u2])

        d4 = Samples([1 2 3 4 5 6 7 8; 8 7 6 5 4 3 2 1])
        c4 = [no_constraint(), no_constraint()]
        name4 = "constrained_MVsampled"
        u4 = ParameterDistribution(d4, c4, name4)

        v = combine_distributions([u1, u2, u3, u4])

        # Tests for sample distribution
        # Note from julia 1.8.5, rand(X,2) != [rand(X,1) rand(X,1)]
        seed = 2020
        testd = MvNormal(zeros(4), 0.1 * I)
        Random.seed!(seed)
        s1 = rand(testd, 1)
        Random.seed!(seed)
        @test sample(u1) == s1

        Random.seed!(seed)
        s1 = [rand(testd, 1) rand(testd, 1) rand(testd, 1)]
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
        s3 = zeros(3, 1)
        s3[1] = rand(Beta(2, 2))
        s3[2:3] = rand(MvNormal(zeros(2), 0.1 * I))
        Random.seed!(seed)
        @test sample(u3) == s3

        Random.seed!(seed)
        s1 = sample(u1, 3)
        s2 = sample(u2, 3)
        s3 = sample(u3, 3)
        s4 = sample(u4, 3)
        Random.seed!(seed)
        s = sample(v, 3)
        @test s == cat([s1, s2, s3, s4]..., dims = 1)

        #Test for logpdf
        @test_throws ErrorException logpdf(u, zeros(ndims(u)))
        x_in_bd = [0.5, 0.5, 0.5]
        Random.seed!(seed)
        # for VectorOfParameterized
        lpdf3 = sum([logpdf(Beta(2, 2), x_in_bd[1])[1], logpdf(MvNormal(zeros(2), 0.1 * I), x_in_bd[2:3])[1]]) #throws deprecated warning without "."     
        Random.seed!(seed)
        @test isapprox(logpdf(u3, x_in_bd) - lpdf3, 0.0; atol = 1e-6)
        @test_throws DimensionMismatch logpdf(u3, [0.5, 0.5])
        # for Parameterized Multivar
        x_in_bd = [0.0, 0.0, 0.0, 0.0]
        @test isapprox(logpdf(u1, x_in_bd) - logpdf(MvNormal(zeros(4), 0.1 * I), x_in_bd)[1], 0.0, atol = 1e-6)
        @test_throws DimensionMismatch logpdf(u1, [1])
        # for Parameterized Univar
        u5 = constrained_gaussian("u5", 3.0, 1.0, -Inf, Inf)
        x_in_bd = 0.0
        @test isapprox(logpdf(u5, x_in_bd) - logpdf(Normal(3.0, 1.0), x_in_bd)[1], 0.0, atol = 1e-6)
        @test_throws DimensionMismatch logpdf(u1, [1, 1])
        @test isapprox(
            logpdf(Parameterized(Normal(3.0, 1.0)), x_in_bd) - logpdf(Normal(3.0, 1.0), x_in_bd)[1],
            0.0,
            atol = 1e-6,
        )
        @test_throws DimensionMismatch logpdf(Parameterized(Normal(3.0, 1.0)), [1, 1])

        #Test for cov, var        
        block_cov = cat([cov(d1), var(d2), cov(d3), cov(d4)]..., dims = (1, 2))
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
        test_d = MvNormal(zeros(4), 0.1 * I)
        test_d3a = Beta(2, 2)
        test_d3b = MvNormal(zeros(2), 0.1 * I)
        d0 = Parameterized(test_d)

        d1 = Parameterized(test_d)
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1, c1, name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10, 15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2, c2, name2)

        d3 = VectorOfParameterized([Beta(2, 2), MvNormal(zeros(2), 0.1 * I)])
        c3 = repeat([no_constraint()], 3)
        name3 = "beta_and_normal"
        u3 = ParameterDistribution(d3, c3, name3)

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

        rng1_new = copy(rng1) # (copy here, as internally u3 samples rng1 sequentially)
        @test sample(copy(rng1), u3) == cat([rand(rng1_new, test_d3a, 1), rand(rng1_new, test_d3b, 1)]..., dims = 1)
        rng1_new = copy(rng1)
        @test sample(copy(rng1), u3, 3) ==
              cat([reshape(rand(rng1_new, test_d3a, 3), :, 3), rand(rng1_new, test_d3b, 3)]..., dims = 1)
        rng1_new = copy(rng1)
        @test sample(copy(rng1), d3) == cat([rand(rng1_new, test_d3a, 1), rand(rng1_new, test_d3b, 1)]..., dims = 1)
        rng1_new = copy(rng1)
        @test sample(copy(rng1), d3, 1) == cat([rand(rng1_new, test_d3a, 1), rand(rng1_new, test_d3b, 1)]..., dims = 1)

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

        rng2_new = copy(rng2)
        @test sample(copy(rng2), u3) == cat([rand(rng2_new, test_d3a, 1), rand(rng2_new, test_d3b, 1)]..., dims = 1)
        rng2_new = copy(rng2)
        @test sample(copy(rng2), u3, 3) ==
              cat([reshape(rand(rng2_new, test_d3a, 3), :, 3), rand(rng2_new, test_d3b, 3)]..., dims = 1)

        # test that optional parameter defaults to Random.GLOBAL_RNG, for all methods.
        # reset the global seed instead of copying the rng object's state
        # Note, from julia 1.8.5 rand(X,2) != [rand(X,1) rand(X,1)]
        rng_seed = 2468
        Random.seed!(rng_seed)
        test_lhs = sample(d0)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        test_lhs = sample(d0, 3)
        Random.seed!(rng_seed)
        @test test_lhs == [rand(test_d, 1) rand(test_d, 1) rand(test_d, 1)]

        Random.seed!(rng_seed)
        test_lhs = sample(u1)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        test_lhs = sample(u1, 3)
        Random.seed!(rng_seed)
        @test test_lhs == [rand(test_d, 1) rand(test_d, 1) rand(test_d, 1)]

        Random.seed!(rng_seed)
        test_lhs = sample(d0)
        Random.seed!(rng_seed)
        @test test_lhs == rand(test_d, 1)

        Random.seed!(rng_seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 1)
        test_lhs = d2.distribution_samples[:, idx]
        Random.seed!(rng_seed)
        @test test_lhs == sample(u2)

        Random.seed!(rng_seed)
        test_lhs = sample(d3)
        Random.seed!(rng_seed)
        @test test_lhs == cat([rand(test_d3a, 1), rand(test_d3b, 1)]..., dims = 1)

    end

    @testset "transform definitions" begin
        tol = 1e-8
        cons = [
            no_constraint(),
            bounded_below(-1.0),
            bounded_below(2.0),
            bounded_above(-1.0),
            bounded_above(2.0),
            bounded(-1.0, 2.0),
        ]
        for con in cons
            for u0 in [-Inf, -5.0, -1.0, 0.0, 1.0, 2.0, 5.0, Inf]
                c0 = con.unconstrained_to_constrained(u0)
                @test ~isnan(c0)
                u1 = con.constrained_to_unconstrained(c0)
                @test ~isnan(u1)
                if isinf(u0)
                    @test u0 == u1
                else
                    @test isapprox(u0, u1, atol = tol)
                end
            end
        end
        cons = [no_constraint(), bounded_below(-1.0), bounded_above(2.0), bounded(-1.0, 2.0)]
        for con in cons
            for c0 in [-1.0, 0.0, 1.0, 2.0]
                @test ~isnan(con.c_to_u_jacobian(c0))
                @test con.c_to_u_jacobian(c0) >= 0.0
                u0 = con.constrained_to_unconstrained(c0)
                @test ~isnan(u0)
                c1 = con.unconstrained_to_constrained(u0)
                @test ~isnan(c1)
                @test isapprox(c0, c1, atol = tol)
            end
        end

        @testset "transform definitions: handle +/-Inf" begin
            for cons in (bounded(-Inf, Inf), bounded_below(-Inf), bounded_above(Inf))
                @test cons.constrained_to_unconstrained(5.0) == 5.0
                @test cons.unconstrained_to_constrained(5.0) == 5.0
            end
            c_hi = bounded_above(5.0)
            cons = bounded(-Inf, 5.0)
            @test cons.unconstrained_to_constrained(10.0) == c_hi.unconstrained_to_constrained(10.0)
            c_lo = bounded_below(-5.0)
            cons = bounded(-5.0, Inf)
            @test cons.unconstrained_to_constrained(-10.0) == c_lo.unconstrained_to_constrained(-10.0)
        end
    end

    @testset "transform functions" begin
        #setup for the tests
        tol = 1e-8
        d1 = Parameterized(MvNormal(zeros(4), 0.1 * I))
        c1 = [no_constraint(), bounded_below(-1.0), bounded_above(0.4), bounded(-0.1, 0.2)]
        n1 = "constrained_mvnormal"
        param_dict1 = Dict("distribution" => d1, "constraint" => c1, "name" => n1)
        u1 = ParameterDistribution(param_dict1)

        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [bounded(10, 15), no_constraint()]
        n2 = "constrained_sampled"
        param_dict2 = Dict("distribution" => d2, "constraint" => c2, "name" => n2)
        u2 = ParameterDistribution(param_dict2)

        param_dict = [param_dict1, param_dict2]
        u = ParameterDistribution(param_dict)

        x_unbd = rand(MvNormal(zeros(6), 3 * I), 1000)  #6 x 1000 
        # Tests for transforms
        x_real_constrained1 = mapslices(x -> transform_unconstrained_to_constrained(u1, x), x_unbd[1:4, :]; dims = 1)
        size(x_real_constrained1)
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

        # vector inputs, for multidim
        u_to_c_vec = transform_unconstrained_to_constrained(u1, vec(x_unbd[1:4, 1]))
        c_to_u_vec = transform_constrained_to_unconstrained(u1, vec(x_real_constrained1[1:4, 1]))
        @test isa(u_to_c_vec, AbstractVector)
        @test isa(c_to_u_vec, AbstractVector)
        @test isapprox(u_to_c_vec, vec(x_real_constrained1[1:4, 1]); atol = tol)
        @test isapprox(vec(x_unbd[1:4, 1]) - c_to_u_vec, zeros(size(vec(x_unbd[1:4, 1]))); atol = tol)

        # vector inputs, for scalar dim and multiple samples
        u4 = ParameterDistribution(
            Dict(
                "distribution" => Parameterized(Normal(2, 3)),
                "constraint" => [bounded_below(4)],
                "name" => "one_dim_bdd_below",
            ),
        )
        x_real_constrained4 = mapslices(x -> transform_unconstrained_to_constrained(u4, x), x_unbd[1, :]; dims = 1)
        u_to_c_vec = transform_unconstrained_to_constrained(u4, vec(x_unbd[1, :]))
        c_to_u_vec = transform_constrained_to_unconstrained(u4, vec(x_real_constrained4))
        u_to_c_real = transform_unconstrained_to_constrained(u4, x_unbd[1, 1])
        c_to_u_real = transform_constrained_to_unconstrained(u4, x_real_constrained4[1, 1])
        @test isa(u_to_c_vec, AbstractVector)
        @test isa(c_to_u_vec, AbstractVector)
        @test isa(u_to_c_real, Real)
        @test isa(c_to_u_real, Real)
        @test isapprox(u_to_c_vec, vec(x_real_constrained4); atol = tol)
        @test isapprox(vec(x_unbd[1, :]) - c_to_u_vec, zeros(size(vec(x_unbd[1, :]))); atol = tol)
        @test isapprox(u_to_c_real, x_real_constrained4[1, 1]; atol = tol)
        @test isapprox(x_unbd[1, 1], c_to_u_real; atol = tol)


        # with transforming samples distributions - using the dict from get_distributions
        d3 = Samples([-10.0 10.0 30.0 -30.0])
        c3 = [bounded_below(0)]
        n3 = "pos_samples"
        param_dict3 = Dict("distribution" => d3, "constraint" => c3, "name" => n3)
        u3 = ParameterDistribution(param_dict3)
        samples_dist = ParameterDistribution([param_dict2, param_dict3]) #combine two samples distributions
        samples_dict_unconstrained = get_distribution(samples_dist) # gives name-> values dict
        samples_dict_constrained = transform_unconstrained_to_constrained(samples_dist, samples_dict_unconstrained)
        samples_dict_unconstrained_again =
            transform_constrained_to_unconstrained(samples_dist, samples_dict_constrained)

        @test isapprox(
            transform_unconstrained_to_constrained(u3, samples_dict_unconstrained[n3]) - samples_dict_constrained[n3],
            zeros(size(samples_dict_constrained[n3]));
            atol = tol,
        )

        @test isapprox(
            transform_unconstrained_to_constrained(u2, samples_dict_unconstrained[n2]) - samples_dict_constrained[n2],
            zeros(size(samples_dict_constrained[n2]));
            atol = tol,
        )

        @test isapprox(
            transform_constrained_to_unconstrained(u3, samples_dict_constrained[n3]) -
            samples_dict_unconstrained_again[n3],
            zeros(size(samples_dict_constrained[n3]));
            atol = tol,
        )

        @test isapprox(
            transform_constrained_to_unconstrained(u2, samples_dict_constrained[n2]) -
            samples_dict_unconstrained_again[n2],
            zeros(size(samples_dict_constrained[n2]));
            atol = tol,
        )

    end

    @testset "constrained_gaussian" begin
        @testset "constrained_gaussian: bounds" begin
            @test_throws DomainError constrained_gaussian("test", 0.0, 1.0, 5.0, -5.0)
            @test_throws DomainError constrained_gaussian("test", -10.0, 1.0, -5.0, 5.0)
            @test_throws DomainError constrained_gaussian("test", 10.0, 1.0, -5.0, 5.0)
            # σ too wide relative to constraint interval
            @test_throws DomainError constrained_gaussian("test", 0.0, 10.0, -1.0, 1000.0)
            @test_throws DomainError constrained_gaussian("test", 0.0, 10.0, -1000.0, 1.0)
            @test_throws DomainError constrained_gaussian("test", 0.0, 10.0, -1.0, 1.0)
            # σ near boundary throws warning
            @test_logs (:warn,) constrained_gaussian("test", 0.54, 0.4, 0, 1) # 0.54 + 1.2*0.4 > 1
            @test_logs (:warn,) constrained_gaussian("test", 0.46, 0.4, 0, 1) # 0.46 - 1.2*0.4 < 1


        end
        @testset "constrained_gaussian: closed form" begin
            μ_c = -5.0
            σ_c = 2.0
            pd = constrained_gaussian("test", μ_c, σ_c, -Inf, Inf)
            d = get_distribution(pd)["test"]
            @test mean(d) == μ_c
            @test std(d) == σ_c
            pd = constrained_gaussian("test", μ_c, σ_c, -Inf, 10.0)
            d = get_distribution(pd)["test"]
            μ_u, σ_u = ParameterDistributions._inverse_lognormal_mean_std(10.0 - μ_c, σ_c)
            @test mean(d) == μ_u
            @test std(d) == σ_u
            pd = constrained_gaussian("test", μ_c, σ_c, -20.0, Inf)
            d = get_distribution(pd)["test"]
            μ_u, σ_u = ParameterDistributions._inverse_lognormal_mean_std(μ_c + 20.0, σ_c)
            @test mean(d) == μ_u
            @test std(d) == σ_u

            #multidimension through repeats
            pd_multi = constrained_gaussian("test", μ_c, σ_c, -20.0, Inf, repeats = 10)
            @test ndims(pd_multi) == 10
            @test all(get_distribution(pd_multi)["test"] .== get_distribution(pd)["test"])
            @test all(get_all_constraints(pd_multi) .== pd.constraint)
            @test get_name(pd_multi) == get_name(pd)

            pd_rep0 = constrained_gaussian("test", μ_c, σ_c, -20.0, Inf, repeats = 0)
            @test pd_rep0 == pd
        end

        @testset "constrained_gaussian: integration" begin
            # verify analytic solutions
            μ_0 = -5.0
            σ_0 = 2.0
            μ_c, σ_c = ParameterDistributions._lognormal_mean_std(μ_0, σ_0)
            μ_u, σ_u = ParameterDistributions._inverse_lognormal_mean_std(μ_c, σ_c)
            @test isapprox(μ_0, μ_u, atol = 1e-7, rtol = 1e-7)
            @test isapprox(σ_0, σ_u, atol = 1e-7, rtol = 1e-7)
            μ_0 = 1.0
            σ_0 = 2.0
            μ_u, σ_u = ParameterDistributions._inverse_lognormal_mean_std(μ_0, σ_0)
            μ_c, σ_c = ParameterDistributions._lognormal_mean_std(μ_u, σ_u)
            @test isapprox(μ_0, μ_c, atol = 1e-7, rtol = 1e-7)
            @test isapprox(σ_0, σ_c, atol = 1e-7, rtol = 1e-7)

            # lognormal case
            cons = bounded(0.0, Inf)
            μ, σ = ParameterDistributions._lognormal_mean_std(0.0, 1.0)
            m, s = ParameterDistributions._mean_std(0.0, 1.0, cons)
            @test isapprox(μ, m, atol = 1e-6, rtol = 1e-6)
            @test isapprox(σ, s, atol = 1e-6, rtol = 1e-6)
            μ, σ = ParameterDistributions._lognormal_mean_std(1.0, 2.0)
            m, s = ParameterDistributions._mean_std(1.0, 2.0, cons)
            @test isapprox(μ, m, atol = 1e-6, rtol = 1e-6)
            @test isapprox(σ, s, atol = 1e-6, rtol = 1e-6)

            # logitnormal
            cons = bounded(0.0, 1.0)
            m, s = ParameterDistributions._mean_std(0.0, 1.0, cons)
            @test isapprox(m, 0.5, atol = 1e-5, rtol = 1e-5)
            @test isapprox(s, 0.20827630, atol = 1e-5, rtol = 1e-5)
            m, s = ParameterDistributions._mean_std(1.0, 2.0, cons)
            @test isapprox(m, 0.64772644, atol = 1e-5, rtol = 1e-5)
            @test isapprox(s, 0.29610580, atol = 1e-5, rtol = 1e-5)

        end

        @testset "constrained_gaussian: optimization" begin
            # lognormal - analytic
            μ_u = 1.0
            σ_u = 2.0
            μ_c, σ_c = ParameterDistributions._lognormal_mean_std(μ_u, σ_u)
            pd = constrained_gaussian("test", μ_c, σ_c, 0.0, Inf)
            d = pd.distribution[1].distribution
            @test isapprox(μ_u, mean(d), atol = 1e-5, rtol = 1e-5)
            @test isapprox(σ_u, std(d), atol = 1e-5, rtol = 1e-5)

            # logitnormal
            μ_c = 0.2
            σ_c = 0.1
            pd = constrained_gaussian("test", μ_c, σ_c, 0.0, 1.0)
            d = pd.distribution[1].distribution
            c = pd.constraint[1]
            m_c, s_c = ParameterDistributions._mean_std(mean(d), std(d), c)
            @test isapprox(m_c, μ_c, atol = 1e-4, rtol = 1e-3)
            @test isapprox(s_c, σ_c, atol = 1e-4, rtol = 1e-3)
            @test isapprox(-1.5047670627292984, mean(d), atol = 1e-4, rtol = 1e-3)
            @test isapprox(0.6474290829043071, std(d), atol = 1e-4, rtol = 1e-3)
        end
    end
end
