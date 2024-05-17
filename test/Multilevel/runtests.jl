using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

Random.seed!(123)
artificial_noise = randn(2)

forward_model(u; level) = begin
    p(x) = u[2]*x + exp(-u[1])*(-x^2/2 + x/2)
    exact_solution = [p(.25); p(.75)]
    exact_solution + u.^2/norm(u.^2) .* artificial_noise / (10 * 2^(level+1))
end

@testset "Multilevel" begin
    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)

    priors = [
        ParameterDistribution(Parameterized(Normal(-3, 1)), no_constraint(), "u1"),
        ParameterDistribution(Parameterized(Normal(105, 5)), no_constraint(), "u2"),
    ]
    prior = combine_distributions(priors)

    y = [27.5; 79.7]
    Γ = 0.01 * I
    N_iter = 10
    lrs = DefaultScheduler(1)

    # Approximate mean-field limit
    println("Approximating mean field")
    N_ens = 200_000
    level = 30
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
    eki = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler=lrs)
    for i in 1:N_iter
        u = get_u_final(eki)
        g_ens = hcat((forward_model(u[:,j]; level) for j in 1:N_ens)...)
        EKP.update_ensemble!(eki, g_ens)
    end
    mean_field_limit_approx_mean = compute_mean(eki, get_u_final(eki))

    # Single-level approximation
    println("Approximating single-level")
    num_avg = 5
    single_level_cost = 2^20
    single_level_errors = map(4:10) do level
        N_ens = floor(Int, single_level_cost / 2^level)
        mean(1:num_avg) do _
            initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
            eki = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler=lrs)
            for i in 1:N_iter
                u = get_u_final(eki)
                g_ens = hcat((forward_model(u[:,j]; level) for j in 1:N_ens)...)
                EKP.update_ensemble!(eki, g_ens)
            end
            norm(mean_field_limit_approx_mean - compute_mean(eki, get_u_final(eki)))
        end
    end

    # Multilevel approximation
    println("Approximating multilevel")
    max_level = 9
    Js = Dict(level => floor(Int, 20 * 2^((max_level - level) * 4/3)) for level in 0:max_level)
    level_scheduler = MultilevelScheduler(Js)
    N_ens = get_N_ens(level_scheduler)
    num_avg = 5
    multilevel_error = mean(1:num_avg) do _
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, get_N_indep(level_scheduler))
        initial_ensemble = transform_noise(level_scheduler, initial_ensemble)
        eki = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng, level_scheduler, scheduler=lrs)
        for i in 1:N_iter
            u = get_u_final(eki)
            g_ens = hcat((forward_model(u[:,j]; level) for (j, level) in zip(1:N_ens, levels(level_scheduler)))...)
            EKP.update_ensemble!(eki, g_ens)
        end
        norm(mean_field_limit_approx_mean - compute_mean(eki, get_u_final(eki)))
    end
    multilevel_cost = reduce(Js; init = 0) do acc, (level, J)
        acc + J * 2^level + (level == 0 ? 0 : J * 2^(level - 1))
    end

    println(multilevel_cost, " ", multilevel_error)
    println(single_level_cost, " ", single_level_errors)
    @test multilevel_cost < 0.5 * single_level_cost
    for single_level_error in single_level_errors
        @test multilevel_error < single_level_error
    end
end
