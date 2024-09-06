using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
const EKP = EnsembleKalmanProcesses

# Read inverse problem definitions
include("../EnsembleKalmanProcess/inverse_problem.jl")


@testset "SparseInversion" begin

    n_obs = 1                   # dimension of synthetic observation from G(u)
    ϕ_star = [-1.0, 2.0]        # True parameters in constrained space
    n_par = size(ϕ_star, 1)
    noise_level = 0.1           # Defining the observation noise level (std)
    N_ens = 20                  # number of ensemble members
    N_iter = 5

    # Test different AbstractMatrices as covariances
    obs_corrmats = [1.0 * I, Matrix(1.0 * I, n_obs, n_obs), Diagonal(Matrix(1.0 * I, n_obs, n_obs))]
    # Test different localizers
    loc_methods = [RBF(2.0), Delta(), NoLocalization()]

    #### Define prior information on parameters assuming independence of parameters
    prior_1 =
        Dict("distribution" => Parameterized(Normal(0.0, 0.25)), "constraint" => bounded(-2.0, 2.0), "name" => "cons_p")
    prior_2 =
        Dict("distribution" => Parameterized(Normal(3.0, 0.5)), "constraint" => no_constraint(), "name" => "uncons_p")
    prior = ParameterDistribution([prior_1, prior_2])
    prior_mean = mean(prior)
    prior_cov = cov(prior)

    # Define a few inverse problems to compare algorithmic performance
    rng_seed = 42234
    rng = Random.MersenneTwister(rng_seed)
    nl_inv_problems = [
        [
            nonlinear_inv_problem_old(ϕ_star, noise_level, n_obs, rng, obs_corrmat = corrmat) for
            corrmat in obs_corrmats
        ]...
        [
            nonlinear_inv_problem_old(
                ϕ_star,
                noise_level,
                n_obs,
                rng,
                obs_corrmat = corrmat,
                add_or_mult_noise = "add",
            ) for corrmat in obs_corrmats
        ]...
    ]

    iters_with_failure = [2]

    # Sparse EKI parameters
    γ = 10.0
    regs = [1e-4, 1e-3]
    uc_idxs = [[1, 2], :]

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
    @test size(initial_ensemble) == (n_par, N_ens)

    threshold_values = [0, 1e-2]
    test_names = ["test", "test_thresholded"]

    for (threshold_value, reg, uc_idx, test_name, inv_problem, loc_method) in
        zip(threshold_values, regs, uc_idxs, test_names, nl_inv_problems, loc_methods)

        y_obs, G, Γy = inv_problem

        process = SparseInversion(γ, threshold_value, uc_idx, reg)
        scheduler = DefaultScheduler(1)
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            deepcopy(process);
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            localization_method = loc_method,
            scheduler = deepcopy(scheduler),
        )
        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            deepcopy(process);
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            scheduler = deepcopy(scheduler),
        )

        # EKI iterations
        u_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        for i in 1:N_iter
            # Check SammpleSuccGauss handler
            params_i = get_ϕ_final(prior, ekiobj)
            push!(u_i_vec, get_u_final(ekiobj))
            g_ens = hcat([G(params_i[:, i]) for i in 1:N_ens]...)
            # Add random failures
            if i in iters_with_failure
                g_ens[:, 1] .= NaN
            end

            push!(g_ens_vec, g_ens)
            if i == 1
                g_ens_t = permutedims(g_ens, (2, 1))
                @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
                EKP.update_ensemble!(ekiobj, g_ens)
            else
                EKP.update_ensemble!(ekiobj, g_ens, Δt_new = get_Δt(ekiobj)[1])
            end
            @test !any(isnan.(params_i))
        end
        push!(u_i_vec, get_u_final(ekiobj))

        @test get_u_prior(ekiobj) == u_i_vec[1]
        @test get_u(ekiobj) == u_i_vec
        @test isequal(get_g(ekiobj), g_ens_vec)
        @test isequal(get_g_final(ekiobj), g_ens_vec[end])
        @test isequal(get_error(ekiobj), ekiobj.error)

        # EKI results: Test if ensemble has collapsed toward the true constrained parameter
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))
        @test eki_final_spread < 2 * eki_init_spread # we wouldn't expect the spread to increase much in any one dimension


        ϕ_final_mean = transform_unconstrained_to_constrained(prior, eki_final_result)
        ϕ_init_mean = transform_unconstrained_to_constrained(prior, eki_init_result)
        @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
        @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2

        # Plot evolution of the EKI particles in constrained space
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, ekiobj, joinpath(@__DIR__, "SEKI_$(test_name).png"))
        end

        # Test other constructors
        @test isa(SparseInversion(γ), SparseInversion)

    end

    ## Repeat first test with several schedulers
    y_obs, G, Γy = nl_inv_problems[1]

    T_end = 3
    schedulers = [
        DefaultScheduler(0.1),
        MutableScheduler(0.1),
        #        DataMisfitController(terminate_at = T_end), # This test can be unstable
    ]
    N_iters = [10, 10]# ..., 20]

    final_ensembles = []
    init_means = []
    final_means = []
    for (scheduler, N_iter) in zip(schedulers, N_iters)

        println("Scheduler: ", nameof(typeof(scheduler)))
        process = SparseInversion(γ, threshold_values[1], uc_idxs[1], regs[1])
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            deepcopy(process);
            rng = copy(rng), #so we get similar performance
            failure_handler_method = SampleSuccGauss(),
            scheduler = scheduler,
        )
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ekiobj)
            g_ens = G(params_i)
            if i == 3
                terminated = EKP.update_ensemble!(ekiobj, g_ens, Δt_new = 0.2)
                #will change Default for 1 step and Mutated for all continuing steps
            else
                terminated = EKP.update_ensemble!(ekiobj, g_ens)
            end
            if !isnothing(terminated)
                break
            end

        end
        push!(init_means, vec(mean(get_u_prior(ekiobj), dims = 2)))
        push!(final_means, vec(mean(get_u_final(ekiobj), dims = 2)))
        # this test is fine so long as N_iter is large enough to hit the termination time
        if isa(scheduler, DataMisfitController)
            if (scheduler.terminate_at, scheduler.on_terminate) == (Float64(T_end), "stop")
                @test sum(get_Δt(ekiobj)) ≈ scheduler.terminate_at
            end
        end

    end
    for i in 1:length(final_means)
        u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
        inv_sqrt_Γy = sqrt(inv(Γy))
        #        @test norm(u_star - final_means[i]) < norm(u_star - init_means[i])
        @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i])))) <
              norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))
    end

end
