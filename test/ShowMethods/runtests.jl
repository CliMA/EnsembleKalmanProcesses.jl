using Test
using Distributions
using LinearAlgebra
using Random
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

const EKP = EnsembleKalmanProcesses

# Helpers
show_str(x) = sprint(show, MIME("text/plain"), x)
summ_str(x) = sprint(summary, x)

function check_show(x, typename)
    s = show_str(x)
    @test occursin(typename, s)
    @test count(==('\n'), s) <= 10
end

function check_summary(x, typename)
    s = summ_str(x)
    @test occursin(typename, s)
    @test !occursin('\n', s)
end

@testset "ShowMethods" begin

    # ── DataContainers ────────────────────────────────────────────────────────

    @testset "DataContainer" begin
        dc = DataContainer(rand(3, 5))
        check_show(dc, "DataContainer")
        check_summary(dc, "DataContainer")
    end

    @testset "PairedDataContainer" begin
        pdc = PairedDataContainer(rand(2, 4), rand(5, 4))
        check_show(pdc, "PairedDataContainer")
        check_summary(pdc, "PairedDataContainer")
    end

    # ── Covariance helpers ────────────────────────────────────────────────────

    @testset "SVDplusD" begin
        A = randn(Random.MersenneTwister(1), 5, 3)
        s = svd(A * A')
        spd = SVDplusD(s, Diagonal(ones(5)))
        check_show(spd, "SVDplusD")
        check_summary(spd, "SVDplusD")
    end

    @testset "DminusTall" begin
        A = randn(Random.MersenneTwister(1), 5, 3)
        s = svd(A * A')
        spd = SVDplusD(s, Diagonal(fill(2.0, 5)))
        dmt = inv_cov(spd)
        check_show(dmt, "DminusTall")
        check_summary(dmt, "DminusTall")
    end

    # ── Observations ─────────────────────────────────────────────────────────

    @testset "Observation" begin
        obs = Observation(Dict("samples" => [1.0, 2.0], "covariances" => 1.0 * I, "names" => "y"))
        check_show(obs, "Observation")
        check_summary(obs, "Observation")
    end

    @testset "FixedMinibatcher" begin
        mb = FixedMinibatcher([[1, 2], [3, 4]])
        check_show(mb, "FixedMinibatcher")
        check_summary(mb, "FixedMinibatcher")
    end

    @testset "RandomFixedSizeMinibatcher" begin
        mb = RandomFixedSizeMinibatcher(2)
        check_show(mb, "RandomFixedSizeMinibatcher")
        check_summary(mb, "RandomFixedSizeMinibatcher")
    end

    @testset "ObservationSeries" begin
        obs1 = Observation(Dict("samples" => [1.0, 2.0], "covariances" => 1.0 * I, "names" => "y"))
        obs2 = Observation(Dict("samples" => [3.0, 4.0], "covariances" => 1.0 * I, "names" => "y"))
        os = ObservationSeries([obs1, obs2])
        check_show(os, "ObservationSeries")
        check_summary(os, "ObservationSeries")
    end

    # ── ParameterDistributions ────────────────────────────────────────────────

    @testset "Parameterized" begin
        p = Parameterized(Normal(0, 1))
        check_show(p, "Parameterized")
        check_summary(p, "Parameterized")
    end

    @testset "Samples" begin
        s = Samples(rand(3, 10))
        check_show(s, "Samples")
        check_summary(s, "Samples")
    end

    @testset "VectorOfParameterized" begin
        vop = VectorOfParameterized([Normal(0, 1), Normal(1, 2)])
        check_show(vop, "VectorOfParameterized")
        check_summary(vop, "VectorOfParameterized")
    end

    @testset "ParameterDistribution" begin
        pd = ParameterDistribution(
            Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "p"),
        )
        check_show(pd, "ParameterDistribution")
        check_summary(pd, "ParameterDistribution")
    end

    @testset "Constraint" begin
        cons = no_constraint()
        check_show(cons, "Constraint")
        check_summary(cons, "Constraint")
        cons_lb = bounded_below(0.0)
        check_show(cons_lb, "Constraint")
        check_summary(cons_lb, "Constraint")
    end

    # ── UpdateGroup ───────────────────────────────────────────────────────────

    @testset "UpdateGroup" begin
        ug = UpdateGroup([1, 2], [1, 3])
        check_show(ug, "UpdateGroup")
        check_summary(ug, "UpdateGroup")
    end

    # ── Accelerators ──────────────────────────────────────────────────────────

    @testset "ConstantNesterovAccelerator" begin
        acc = ConstantNesterovAccelerator()
        check_show(acc, "ConstantNesterovAccelerator")
        check_summary(acc, "ConstantNesterovAccelerator")
    end

    @testset "FirstOrderNesterovAccelerator" begin
        acc = FirstOrderNesterovAccelerator()
        check_show(acc, "FirstOrderNesterovAccelerator")
        check_summary(acc, "FirstOrderNesterovAccelerator")
    end

    @testset "NesterovAccelerator" begin
        acc = NesterovAccelerator()
        check_show(acc, "NesterovAccelerator")
        check_summary(acc, "NesterovAccelerator")
    end

    # ── LearningRateSchedulers ────────────────────────────────────────────────

    @testset "DataMisfitController" begin
        dmc = DataMisfitController()
        check_show(dmc, "DataMisfitController")
        check_summary(dmc, "DataMisfitController")
    end

    # ── Process types ─────────────────────────────────────────────────────────

    @testset "Inversion" begin
        inv = Inversion()
        check_show(inv, "Inversion")
        check_summary(inv, "Inversion")
    end

    @testset "TransformInversion" begin
        ti = TransformInversion()
        check_show(ti, "TransformInversion")
        check_summary(ti, "TransformInversion")
    end

    @testset "Sampler" begin
        prior = ParameterDistribution(
            Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "p"),
        )
        samp = Sampler(prior)
        check_show(samp, "Sampler")
        check_summary(samp, "Sampler")
    end

    @testset "GaussNewtonInversion" begin
        prior = ParameterDistribution(
            Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "p"),
        )
        gni = GaussNewtonInversion(prior)
        check_show(gni, "GaussNewtonInversion")
        check_summary(gni, "GaussNewtonInversion")
    end

    @testset "SparseInversion" begin
        si = SparseInversion(0.1)
        check_show(si, "SparseInversion")
        check_summary(si, "SparseInversion")
    end

    @testset "Unscented" begin
        u0_mean = zeros(2)
        uu0_cov = Matrix(1.0I, 2, 2)
        uki = Unscented(u0_mean, uu0_cov)
        check_show(uki, "Unscented")
        check_summary(uki, "Unscented")
    end

    @testset "TransformUnscented" begin
        u0_mean = zeros(2)
        uu0_cov = Matrix(1.0I, 2, 2)
        tuki = TransformUnscented(u0_mean, uu0_cov)
        check_show(tuki, "TransformUnscented")
        check_summary(tuki, "TransformUnscented")
    end

    # ── EnsembleKalmanProcess ─────────────────────────────────────────────────

    @testset "EnsembleKalmanProcess" begin
        rng = Random.MersenneTwister(42)
        prior = ParameterDistribution(
            Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "p"),
        )
        N_ens = 10
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        y_obs = [1.0]
        Γ = Matrix(1.0I, 1, 1)
        ekpobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γ, Inversion())
        check_show(ekpobj, "EnsembleKalmanProcess")
        check_summary(ekpobj, "EnsembleKalmanProcess")
    end

end
