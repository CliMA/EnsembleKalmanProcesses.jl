using Test
using Distributions
using LinearAlgebra
using Random
using StatsBase

using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.PlotRecipes

if TEST_PLOT_OUTPUT
    @testset "PlotRecipes" begin
        @testset "Plot ParameterDistribution" begin
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

            # PDTs
            plt1 = plot(d1)
            savefig(plt1, joinpath(@__DIR__, name1 * ".png"))
            plt2 = plot(d2)
            savefig(plt2, joinpath(@__DIR__, name2 * ".png"))
            plt3 = plot(d3)
            savefig(plt3, joinpath(@__DIR__, name3 * ".png"))

            # full param dist
            plt_constrained = plot(u)
            savefig(plt_constrained, joinpath(@__DIR__, "full_dist_constrained.png"))
            plt_unconstrained = plot(u, constrained = false)
            savefig(plt_unconstrained, joinpath(@__DIR__, "full_dist_unconstrained.png"))
            # just a redundant test to show that the plots were created
            @test 1 == 1
        end
    end
end
