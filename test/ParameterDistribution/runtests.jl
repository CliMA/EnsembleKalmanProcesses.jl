using Test
using Distributions
using StatsBase
using Random

using EnsembleKalmanProcesses.ParameterDistributionStorage

@testset "ParameterDistributionStorage" begin
    @testset "ParameterDistributionType" begin
        # Tests for the ParameterDistributionType
        d = Parameterized(Gamma(2.0, 0.8))
        @test d.distribution == Gamma(2.0,0.8)
        
        d = Samples([1 2 3; 4 5 6])
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1 4; 2 5; 3 6]; params_are_columns=false)
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6])
        @test d.distribution_samples == [1 2 3 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6]; params_are_columns=false)
        @test d.distribution_samples == reshape([1, 2, 3, 4, 5, 6],:,1)
        
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

        
        c3 = bounded(-0.1,0.2)
        @test isapprox(c3.constrained_to_unconstrained(0.0) - (log( (0.0 - -0.1) / (0.2 - 0.0))) , 0 )
        @test isapprox(c3.unconstrained_to_constrained(1.0) - ((0.2 * exp(1.0) + -0.1) / (exp(1.0) + 1)) , 0)
        @test_throws DomainError bounded(0.2,-0.1)

        #an example with user defined invertible transforms
        c_to_u = (x -> 3 * x + 14)
        u_to_c = (x -> (x - 14) / 3) 
        
        c4 = Constraint(c_to_u, u_to_c)
        @test isapprox( c4.constrained_to_unconstrained(5.0) - c_to_u(5.0) , 0)
        @test isapprox( c4.unconstrained_to_constrained(5.0) - u_to_c(5.0) , 0)
        

    end

    @testset "ParameterDistribution" begin

        d = Parameterized(Normal(0,1))
        c = no_constraint()
        name = "unconstrained_normal"
        u = ParameterDistribution(d,c,name)
        @test u.distributions == [d]
        @test u.constraints == [c]
        @test u.names == [name]

        # Tests for the ParameterDistribution
        d = Parameterized(MvNormal(4,0.1))
        c = [no_constraint(),
             bounded_below(-1.0),
             bounded_above(0.4),
             bounded(-0.1,0.2)]
      
        name = "constrained_mvnormal"
        u = ParameterDistribution(d,c,name)
        @test u.distributions == [d]
        @test u.constraints == c
        @test u.names == [name]
        @test_throws DimensionMismatch ParameterDistribution(d,c[1:3],name)
        @test_throws DimensionMismatch ParameterDistribution(d,c, [name,"extra_name"])
        
        # Tests for the ParameterDistribution
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [no_constraint(),
              bounded_below(-1.0),
              bounded_above(0.4),
              bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [bounded(10,15),
              no_constraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])
        @test u.distributions == [d1,d2]
        @test u.constraints == cat([c1,c2]...,dims=1)
        @test u.names == [name1,name2]
    end

    @testset "getter functions" begin
        # setup for the tests:
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [no_constraint(),
              bounded_below(-1.0),
              bounded_above(0.4),
              bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1 2 3 4])
        c2 = [bounded(10,15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])

        # Test for get_dimension(s)
        @test get_total_dimension(u1) == 4
        @test get_total_dimension(u2) == 1
        @test get_total_dimension(u) == 5
        @test get_dimensions(u1) == [4]
        @test get_dimensions(u2) == [1]
        @test get_dimensions(u) == [4,1]
        
        # Tests for get_name
        @test get_name(u1) == [name1]
        @test get_name(u) == [name1, name2]

        # Tests for get_n_samples
        @test typeof(get_n_samples(u)[name1]) <: String
        @test get_n_samples(u)[name2] == 4
        
        # Tests for get_distribution
        @test get_distribution(d1) == MvNormal(4,0.1)
        @test get_distribution(u1)[name1] == MvNormal(4,0.1)
        @test typeof(get_distribution(d2)) == Array{Int64, 2}
        @test typeof(get_distribution(u2)[name2]) == Array{Int64, 2}
        
        d = get_distribution(u)
        @test d[name1] == MvNormal(4,0.1)
        @test typeof(d[name2]) == Array{Int64, 2}

        # Test for get_all_constraints
        @test get_all_constraints(u) == cat([c1,c2]...,dims=1)
    end

    @testset "statistics functions" begin

        # setup for the tests:
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [no_constraint(),
              bounded_below(-1.0),
              bounded_above(0.4),
              bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)

        d2 = Samples([1 2 3 4])
        c2 = [bounded(10,15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        d3 = Parameterized(Beta(2,2))
        c3 = [no_constraint()]
        name3 = "unconstrained_beta"
        u3 = ParameterDistribution(d3,c3,name3)

        u = ParameterDistribution([d1,d2],[c1,c2],[name1,name2])

        d4 = Samples([1 2 3 4 5 6 7 8; 8 7 6 5 4 3 2 1])
        c4 = [no_constraint(),
              no_constraint()]
        name4 = "constrained_MVsampled"
        v = ParameterDistribution([d1,d2,d3,d4],[c1,c2,c3,c4],[name1,name2,name3,name4])
        
        # Tests for sample distribution
        seed=2020
        Random.seed!(seed)
        s1 = rand(MvNormal(4,0.1),1)
        Random.seed!(seed)
        @test sample_distribution(u1) == s1

        Random.seed!(seed)
        s1 = rand(MvNormal(4,0.1),3)
        Random.seed!(seed)
        @test sample_distribution(u1,3) == s1
        
        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]),1) 
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample_distribution(u2) == s2

        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 3) 
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample_distribution(u2,3) == s2
        
        Random.seed!(seed)
        s1 = sample_distribution(u1,3)
        s2 = sample_distribution(u2,3)
        Random.seed!(seed)
        s = sample_distribution(u,3)
        @test s == cat([s1,s2]...,dims=1)

        #Test for get_logpdf
        @test_throws ErrorException get_logpdf(u,zeros(get_total_dimension(u))) 
        x_in_bd = [0.5]
        Random.seed!(seed)
        lpdf3 = logpdf.(Beta(2,2),x_in_bd)[1] #throws deprecated warning without "."
        Random.seed!(seed)
        @test isapprox(get_logpdf(u3,x_in_bd) - lpdf3 , 0.0; atol=1e-6)
        @test_throws DimensionMismatch get_logpdf(u3, [0.5,0.5])

        #Test for get_cov, get_var        
        block_cov = cat([get_cov(d1),get_var(d2),get_var(d3),get_cov(d4)]..., dims=(1,2)) 
        @test isapprox(get_cov(v) - block_cov, zeros(get_total_dimension(v),get_total_dimension(v)); atol=1e-6)
        #Test for get_mean
        means = cat([get_mean(d1), get_mean(d2), get_mean(d3), get_mean(d4)]...,dims=1)
        @test isapprox(get_mean(v) - means, zeros(get_total_dimension(v)); atol=1e-6)
        
    end

    @testset "transform functions" begin
        #setup for the tests
        tol = 1e-8
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [no_constraint(),
              bounded_below(-1.0),
              bounded_above(0.4),
              bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [bounded(10,15),
              no_constraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])

        x_unbd = rand(MvNormal(6,3), 1000)  #6 x 1000 
        # Tests for transforms
        x_real_constrained1 = mapslices(x -> transform_unconstrained_to_constrained(u1,x),x_unbd[1:4,:]; dims=1)
        @test isapprox(x_unbd[1:4,:] - mapslices(x -> transform_constrained_to_unconstrained(u1,x),
            x_real_constrained1; dims=1), zeros(size(x_unbd[1:4,:])); atol=tol)

        x_real_constrained2 = mapslices(x -> transform_unconstrained_to_constrained(u2,x), x_unbd[5:6,:]; dims=1) 
        @test isapprox(x_unbd[5:6,:] - mapslices(x -> transform_constrained_to_unconstrained(u2,x),
            x_real_constrained2; dims=1), zeros(size(x_unbd[5:6,:])); atol=tol)
        
        x_real = mapslices(x -> transform_unconstrained_to_constrained(u,x), x_unbd; dims=1)
        x_unbd_tmp = mapslices(x -> transform_constrained_to_unconstrained(u,x), x_real; dims=1)
        @test isapprox(x_unbd - x_unbd_tmp,zeros(size(x_unbd)); atol=tol)
        
        # Tests transforms for other input structures
        @test isapprox(transform_unconstrained_to_constrained(u1,x_unbd[1:4,:]), x_real_constrained1; atol=tol)
        @test isapprox(x_unbd[1:4,:] - transform_constrained_to_unconstrained(u1, x_real_constrained1),
            zeros(size(x_unbd[1:4,:])); atol=tol)
        @test isapprox(transform_unconstrained_to_constrained(u2,x_unbd[5:6,:]), x_real_constrained2; atol=tol)
        @test isapprox(x_unbd[5:6,:] - transform_constrained_to_unconstrained(u2, x_real_constrained2),
            zeros(size(x_unbd[5:6,:])); atol=tol)
        @test isapprox(transform_unconstrained_to_constrained(u2,[x_unbd[5:6,:], x_unbd[5:6,:]]),
            [x_real_constrained2, x_real_constrained2]; atol=tol)
    end
end
