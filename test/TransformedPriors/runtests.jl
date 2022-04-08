using Test
using Distributions, StatsBase, Random, LinearAlgebra
using QuadGK, Optim
using SpecialFunctions

using EnsembleKalmanProcesses.ParameterDistributions
m = ParameterDistributions # abbreviate for testing private methods

@testset "TransformedPriors" begin
    @testset "Matrix minors" begin
        # test minor indexing
        @test length(1:5) == length(m.OmitOneRange(1,5,3)) + 1
        @test length(3:3) == length(m.OmitOneRange(3,3,3)) + 1
        @test length(3:10) == length(m.OmitOneRange(3,10,3)) + 1
        @test length(3:10) == length(m.OmitOneRange(3,10,7)) + 1
        @test length(3:10) == length(m.OmitOneRange(3,10,10)) + 1

        @test [j for j in m.OmitOneRange(1,5,2)] == [1, 3, 4, 5]
        @test [j for j in m.OmitOneRange(2,7,2)] == [3, 4, 5, 6, 7]
        @test [j for j in m.OmitOneRange(1,5,nothing)] == [1, 2, 3, 4, 5]
        @test [j for j in m.OmitOneRange(2,7,nothing)] == [2, 3, 4, 5, 6, 7]

        A = reshape(1:9, (3,3))
        @test m._submatrix(A, 1, 1) == [[5 8]; [6 9]] 
        @test m._submatrix(A, 2, 3) == [[1 4]; [3 6]] 
    end


    @testset "Quantiles" begin
        @test isapprox(m.quantile(0.5), 0.0)
        @test isapprox(m.quantile(0.25), -0.674490, atol=1e-6)
    end

    @testset "3-pt transforms" begin
        # test 3-pt transform, 1
        t = m.get_xform_coeffs(2., -1., 5.)

        @test isapprox(m.z_from_w(m.w_from_z(3.3, t), t), 3.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(0.3, t), t), 0.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(9.3, t), t), 9.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(-7.3, t), t), -7.3, atol=1e-6)

        @test isapprox(m.z_from_w(1e100, t),   5.0, atol=1e-6)
        @test isapprox(m.z_from_w(1.0, t),     2.0, atol=1e-6)
        @test isapprox(m.z_from_w(0.0, t),    -1.0, atol=1e-6)

        # 3-pt: verify correct limits of different methods as min or max -> Infinity
        t1 = m.get_xform_coeffs(2., -1.0e8, 5.)
        t2 = m.get_xform_coeffs(2., -Inf, 5.)
        @test isapprox(m.w_from_z(4.0, t1), m.w_from_z(4.0, t2), atol=1e-6)
        @test isapprox(m.w_from_z(-8.0, t1), m.w_from_z(-8.0, t2), atol=1e-6)

        t1 = m.get_xform_coeffs(2., -1., 1.0e8)
        t2 = m.get_xform_coeffs(2., -1., Inf)
        @test isapprox(m.w_from_z(8.0, t1), m.w_from_z(8.0, t2), atol=1e-6)
        @test isapprox(m.w_from_z(-0.5, t1), m.w_from_z(-0.5, t2), atol=1e-6)
    end

    @testset "5-pt transforms" begin
        # test 5-pt transform, 1
        t = m.get_xform_coeffs([1., 2., 3.], -1., 5.)

        @test isapprox(m.z_from_w(m.w_from_z(3.3, t), t), 3.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(0.3, t), t), 0.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(9.3, t), t), 9.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(-7.3, t), t), -7.3, atol=1e-6)

        exp_zp = exp(m.quantile(0.25))
        @test isapprox(m.z_from_w(1e100, t),   5.0, atol=1e-6)
        @test isapprox(m.z_from_w(1/exp_zp, t), 3.0, atol=1e-6)
        @test isapprox(m.z_from_w(1.0, t),     2.0, atol=1e-6)
        @test isapprox(m.z_from_w(exp_zp, t),   1.0, atol=1e-6)
        @test isapprox(m.z_from_w(0.0, t),    -1.0, atol=1e-6)

        # test 5-pt transform, 2
        t = m.get_xform_coeffs([-0.5, 2.0, 4.5], -1.0, 5.0)

        @test isapprox(m.z_from_w(m.w_from_z(3.3, t), t), 3.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.z_from_w(m.w_from_z(0.3, t), t), 0.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(9.3, t), t), 9.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(4.3, t), t), 4.3, atol=1e-6)
        @test isapprox(m.w_from_z(m.z_from_w(-7.3, t), t), -7.3, atol=1e-6)

        exp_zp = exp(m.quantile(0.25))
        @test isapprox(m.z_from_w(1e100, t),   5.0, atol=1e-6)
        @test isapprox(m.z_from_w(1/exp_zp, t), 4.5, atol=1e-6)
        @test isapprox(m.z_from_w(1.0, t),     2.0, atol=1e-6)
        @test isapprox(m.z_from_w(exp_zp, t),  -0.5, atol=1e-6)
        @test isapprox(m.z_from_w(0.0, t),    -1.0, atol=1e-6)

        # 5-pt: verify correct limits of different methods as min or max -> Infinity
        t1 = m.get_xform_coeffs([1., 2., 3.], -1.0e8, 5.)
        t2 = m.get_xform_coeffs([1., 2., 3.], -Inf, 5.)
        @test isapprox(m.w_from_z(4.0, t1), m.w_from_z(4.0, t2), atol=1e-6)
        @test isapprox(m.w_from_z(-8.0, t1), m.w_from_z(-8.0, t2), atol=1e-6)

        t1 = m.get_xform_coeffs([1., 2., 3.], -1., 1.0e8)
        t2 = m.get_xform_coeffs([1., 2., 3.], -1., Inf)
        @test isapprox(m.w_from_z(8.0, t1), m.w_from_z(8.0, t2), atol=1e-6)
        @test isapprox(m.w_from_z(-0.5, t1), m.w_from_z(-0.5, t2), atol=1e-6)

        t1 = m.get_xform_coeffs([1., 2., 3.], -1.0e8, 1.0e8)
        t2 = m.get_xform_coeffs([1., 2., 3.], -Inf, Inf)
        @test isapprox(m.w_from_z(8.0, t1), m.w_from_z(8.0, t2), atol=1e-6)
        @test isapprox(m.w_from_z(-8.0, t1), m.w_from_z(-8.0, t2), atol=1e-6)
    end
end
