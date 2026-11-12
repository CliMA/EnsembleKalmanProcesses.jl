using Distributions
using LinearAlgebra
using Random
using Statistics
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses

@testset "Localization" begin

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)

    # Linear problem with d == p >> N_ens - Section 6.1 of Tong and Morzfeld (2022)
    G(u) = u
    N_enss = [10, 5] #SECNice requires test for N<6 and N>=6
    mask = [repeat([1], 8)..., 2] # will do N_ens =10 for 8 exp then N_ens=5
    p = 50
    N_iter = 20
    # Generate random truth
    y = 10.0 * rand(p)
    Γ = 1.0 * I

    #### Define prior information on parameters
    priors = ParameterDistribution[] #empty PD-array
    for i in 1:p
        push!(priors, ParameterDistribution(Parameterized(Normal(0.0, 0.5)), no_constraint(), string("u", i)))
    end
    prior = combine_distributions(priors)

    scheduler = DefaultScheduler(1)
    # Solve problem without localization
    nonlocalized_errors = []
    for N_ens in N_enss
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        ekiobj_vanilla = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y,
            Γ,
            Inversion();
            rng = rng,
            localization_method = NoLocalization(),
            scheduler = scheduler,
        )
        for i in 1:N_iter
            g_ens_vanilla = G(get_u_final(ekiobj_vanilla))
            EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla)
        end
        push!(nonlocalized_errors, get_error(ekiobj_vanilla)[end])
    end

    # Test different localizers
    loc_methods = [
        Delta(),
        RBF(1.0),
        RBF(0.1),
        BernoulliDropout(0.1),
        SEC(10.0),
        SECFisher(),
        SEC(1.0, 0.1),
        SECNice(),
        SECNice(),
    ]

    for (mask_val, loc_method) in zip(mask, loc_methods)
        N_ens = N_enss[mask_val]
        nonlocalized_error = nonlocalized_errors[mask_val]

        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y,
            Γ,
            Inversion();
            rng = rng,
            localization_method = loc_method,
            scheduler = scheduler,
        )
        @test isa(get_localizer(ekiobj), Localizer)

        for i in 1:N_iter
            g_ens = G(get_u_final(ekiobj))
            EKP.update_ensemble!(ekiobj, g_ens, deterministic_forward_map = true)
        end

        # Check for expansion in some dimension
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))
        @test eki_final_spread < 2 * eki_init_spread


        # Test that localized version does better in the setting p >> N_ens
        @test get_error(ekiobj)[end] < nonlocalized_error

        # Test Schur product theorem
        u_final = get_u_final(ekiobj)
        g_final = get_g_final(ekiobj)
        cov_est = cov([u_final; g_final], dims = 2, corrected = false)
        # The arguments for the localizer
        T, p, d, J = (eltype(g_final), size(u_final, 1), size(g_final, 1), size(u_final, 2))
        cov_localized = ekiobj.localizer.localize(cov_est, T, p, d, J)
        @test rank(cov_est) < rank(cov_localized)
        # Test localization getter method
        @test isa(loc_method, EKP.get_localizer_type(ekiobj))
    end


end

@testset "SECNice std_corrs formula matches the Monte Carlo sampling std of a correlation" begin
    # std(r|ρ) must be even in ρ (mirror-symmetric under ρ → -ρ): pins the
    # (1 - ρ²)/√N formula against an empirical estimate for ρ < 0, ρ = 0, ρ > 0.
    rng = Random.MersenneTwister(42)
    N_ens = 20
    n_trials = 5000
    for ρ in (-0.9, 0.0, 0.5)
        Σ = [1.0 ρ; ρ 1.0]
        corrs = zeros(n_trials)
        for k in 1:n_trials
            X = rand(rng, MvNormal(zeros(2), Σ), N_ens)
            corrs[k] = cor(X[1, :], X[2, :])
        end
        empirical_std = std(corrs)
        formula_std = (1 - ρ^2) / sqrt(N_ens)
        @test isapprox(formula_std, empirical_std, rtol = 0.25)
    end
end

@testset "SEC-family localizers handle a zero-variance row without NaN/Inf" begin
    # A row that is constant across the ensemble has exactly zero sample variance;
    # the correlation-matrix decomposition must not divide by that zero.
    rng = Random.MersenneTwister(3)
    n = 4
    N_ens = 30
    ens = randn(rng, n, N_ens)
    ens[2, :] .= 5.0
    cov_est = cov(ens, dims = 2, corrected = false)
    p, d, J = 3, 1, N_ens
    for loc in (Localizer(SEC(1.0), J), Localizer(SECFisher(), J), Localizer(SECNice(), J))
        cov_localized = loc.localize(cov_est, Float64, p, d, J)
        @test all(isfinite, cov_localized)
        @test cov_localized[2, :] == cov_est[2, :]
        @test cov_localized[:, 2] == cov_est[:, 2]
    end
end
