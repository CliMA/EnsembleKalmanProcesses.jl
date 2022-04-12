using Distributions
using LinearAlgebra
using Random
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
    N_ens = 10
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
    #prior_distns = repeat([Parameterized(Normal(0.0, 0.5))], p)
    #constraints = repeat([[no_constraint()]], p)
    #prior_names = [string("u_", i) for i in 1:p]
    #prior = ParameterDistribution(prior_distns, constraints, prior_names)

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    # Solve problem without localization
    ekiobj_vanilla = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)
    for i in 1:N_iter
        g_ens_vanilla = G(get_u_final(ekiobj_vanilla))
        EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla)
    end
    nonlocalized_error = get_error(ekiobj_vanilla)[end]

    # Test different localizers
    loc_methods = [Delta(), RBF(1.0), RBF(0.1), BernoulliDropout(0.1)]

    for loc_method in loc_methods
        ekiobj =
            EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = loc_method)
        @test isa(ekiobj.localizer, Localizer)

        for i in 1:N_iter
            g_ens = G(get_u_final(ekiobj))
            EKP.update_ensemble!(ekiobj, g_ens, deterministic_forward_map = true)
        end

        # Test that localized version does better in the setting p >> N_ens
        @test get_error(ekiobj)[end] < nonlocalized_error

        # Test Schur product theorem
        u_final = get_u_final(ekiobj)
        g_final = get_g_final(ekiobj)
        cov_ug = cov(u_final, g_final, dims = 2, corrected = false)
        kernel = ekiobj.localizer.kernel
        @test rank(cov_ug) < rank(kernel .* cov_ug)
    end

end
