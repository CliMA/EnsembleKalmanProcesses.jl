using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses



# Read inverse problem definitions
include("inverse_problem.jl")

n_obs = 30                  # dimension of synthetic observation from G(u)
ϕ_star = [-1.0, 2.0]        # True parameters in constrained space
n_par = size(ϕ_star, 1)
noise_level = 0.1           # Defining the observation noise level (std)
N_ens = 50                  # number of ensemble members
N_iter = 20

# Test different AbstractMatrices as covariances
obs_corrmats = [I, Matrix(I, n_obs, n_obs), Diagonal(Matrix(I, n_obs, n_obs))]
# Test different localizers
loc_methods = [RBF(2.0), Delta(), NoLocalization(), NoLocalization()]

#### Define prior information on parameters assuming independence of cons_p and uncons_p
prior_1 = Dict("distribution" => Parameterized(Normal(0.0, 0.5)), "constraint" => bounded(-2, 2), "name" => "cons_p")
prior_2 = Dict("distribution" => Parameterized(Normal(3.0, 0.5)), "constraint" => no_constraint(), "name" => "uncons_p")
prior = ParameterDistribution([prior_1, prior_2])
prior_mean = mean(prior)
prior_cov = cov(prior)

# Define a few inverse problems to compare algorithmic performance
rng_seed = 42
rng = Random.MersenneTwister(rng_seed)
# Random linear forward map
inv_problems = [
    linear_inv_problem(ϕ_star, noise_level, n_obs, rng; obs_corrmat = corrmat, return_matrix = true) for
    corrmat in obs_corrmats
]
n_lin_inv_probs = length(inv_problems)
nl_inv_problems = [
    (nonlinear_inv_problem(ϕ_star, noise_level, n_obs, rng, obs_corrmat = obs_corrmats[3])..., nothing),
    (
        nonlinear_inv_problem(
            ϕ_star,
            noise_level,
            n_obs,
            rng,
            obs_corrmat = obs_corrmats[3],
            add_or_mult_noise = "add",
        )...,
        nothing,
    ),
]
inv_problems = [inv_problems..., nl_inv_problems...]
@testset "Inverse problem definition" begin

    rng = Random.MersenneTwister(rng_seed)

    y_obs, G, Γ, A = linear_inv_problem(ϕ_star, noise_level, n_obs, rng; return_matrix = true)

    # Test dimensionality
    @test size(ϕ_star) == (n_par,)
    @test size(A * ϕ_star) == (n_obs,)
    @test size(y_obs) == (n_obs,)

    # sum(y-G)^2 ~ n_obs*noise_level^2
    @test isapprox(norm(y_obs .- A * ϕ_star)^2 - n_obs * noise_level^2, 0; atol = 0.05)
end


@testset "LearningRateSchedulers" begin
    # Default
    Δt = 3
    dlrs1 = EKP.DefaultScheduler()
    @test dlrs1.Δt_default == Float64(1)
    dlrs2 = EKP.DefaultScheduler(Δt)
    @test dlrs2.Δt_default == Float64(Δt)

    #Mutable
    mlrs1 = EKP.MutableScheduler()
    @test mlrs1.Δt_mutable == Float64[1]
    mlrs2 = EKP.MutableScheduler(Δt)
    @test mlrs2.Δt_mutable == Float64[Δt]

    # EKSStable 
    ekslrs1 = EKP.EKSStableScheduler()
    @test ekslrs1.numerator == Float64(1)
    @test ekslrs1.nugget == Float64(eps())

    num = 3
    nug = 0.0001
    ekslrs2 = EKP.EKSStableScheduler(num, nug)
    @test ekslrs2.numerator == Float64(3)
    @test ekslrs2.nugget == Float64(0.0001)

    num = Float32(3)
    nug = Float32(0.0001)
    ekslrs2 = EKP.EKSStableScheduler(num, nug)
    @test ekslrs2.numerator == Float32(3)
    @test ekslrs2.nugget == Float32(0.0001)

    num = Float32(3)
    nug = Float64(0.0001)
    ekslrs2 = EKP.EKSStableScheduler(num, nug)
    @test ekslrs2.numerator == Float64(3)
    @test ekslrs2.nugget == Float64(0.0001)

    # DataMistfitController
    # DMC has no user parameters, and gets initialized during initial update
    dmclrs1 = EKP.DataMisfitController()
    @test typeof(dmclrs1.iteration) == Vector{Int}
    @test length(dmclrs1.iteration) == 0
    @test typeof(dmclrs1.inv_sqrt_noise) == Vector{Matrix{Float64}}
    @test length(dmclrs1.inv_sqrt_noise) == 0
    @test dmclrs1.terminate_at == Float64(1)
    @test dmclrs1.on_terminate == "stop"
    dmclrs2 = EKP.DataMisfitController(terminate_at = 7, on_terminate = "continue")
    @test dmclrs2.on_terminate == "continue"
    @test dmclrs2.terminate_at == Float64(7)
    dmclrs3 = EKP.DataMisfitController(on_terminate = "continue_fixed")
    @test dmclrs3.on_terminate == "continue_fixed"

    # build EKP and eki objects
    # Get an inverse problem
    y_obs, G, Γy, _ = inv_problems[end] #additive noise inv problem
    rng = Random.MersenneTwister(rng_seed)
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    ekiobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())
    eksobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior))

    @test ekiobj.scheduler == DefaultScheduler{Float64}(1.0)
    @test eksobj.scheduler == EKSStableScheduler{Float64}(1.0, eps())

    #test
    processes = [
        Inversion(),
        #        Unscented(prior), TO BE UNCOMMENTED WHEN UKI BUG-FIXED
        #Sparse inversion tests in test/SparseInversion/runtests.jl
    ]
    T_end = 3 # (this could fail a test if N_iters is not enough to reach T_end)
    for process in processes
        schedulers = [
            DefaultScheduler(0.05),
            MutableScheduler(0.05),
            DataMisfitController(terminate_at = T_end),
            DataMisfitController(on_terminate = "continue"),
            DataMisfitController(on_terminate = "continue_fixed"),
        ]
        N_iters = [40, 40, 40, 40, 40]
        init_means = []
        final_means = []

        for (scheduler, N_iter) in zip(schedulers, N_iters)
            println("Scheduler: ", nameof(typeof(scheduler)))
            if !(nameof(typeof(process)) == Symbol(Unscented))
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_ensemble,
                    y_obs,
                    Γy,
                    process,
                    rng = copy(rng),
                    scheduler = scheduler,
                )
            else #no initial ensemble for UKI
                ekpobj = EKP.EnsembleKalmanProcess(y_obs, Γy, process, rng = copy(rng), scheduler = scheduler)
            end
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)
                g_ens = G(params_i)
                if i == 3
                    terminated = EKP.update_ensemble!(ekpobj, g_ens, Δt_new = 0.1)
                    #will change Default for 1 step and Mutated for all continuing steps
                else
                    terminated = EKP.update_ensemble!(ekpobj, g_ens)
                end
                if !isnothing(terminated)
                    break
                end
            end
            push!(init_means, vec(mean(get_u_prior(ekpobj), dims = 2)))
            push!(final_means, vec(mean(get_u_final(ekpobj), dims = 2)))

            # this test is fine so long as N_iter is large enough to hit the termination time
            if nameof(typeof(scheduler)) == DataMisfitController
                if (scheduler.terminate_at, scheduler.on_terminate) == (Float64(T_end), "stop")
                    @test sum(ekpobj.Δt) ≈ scheduler.terminate_at
                end
            end
        end
        for i in 1:length(final_means)
            u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
            inv_sqrt_Γy = sqrt(inv(Γy))
            #            @test norm(u_star - final_means[i]) < norm(u_star - init_means[i])
            @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i])))) <
                  norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))

        end
    end
end

@testset "EnsembleKalmanSampler" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
    @test size(initial_ensemble) == (n_par, N_ens)

    # Global scope to compare against EKI
    global eks_final_results = []
    global eksobjs = []

    # Test EKS for different inverse problem
    for (i_prob, inv_problem) in enumerate(inv_problems)

        # Get inverse problem
        y_obs, G, Γy, _ = inv_problem

        eksobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior); rng = rng)

        params_0 = get_u_final(eksobj)
        g_ens = G(params_0)
        g_ens_t = permutedims(g_ens, (2, 1))

        @test size(g_ens) == (n_obs, N_ens)
        @test_throws DimensionMismatch EKP.update_ensemble!(eksobj, g_ens_t)

        # EKS iterations
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, eksobj)
            g_ens = G(params_i)
            EKP.update_ensemble!(eksobj, g_ens)
        end
        # Collect mean initial parameter for comparison
        initial_guess = vec(mean(get_u_prior(eksobj), dims = 2))
        # Collect mean final parameter as the solution
        eks_final_result = get_u_mean_final(eksobj)

        @test initial_guess == get_u_mean(eksobj, 1)
        @test eks_final_result == vec(mean(get_u_final(eksobj), dims = 2))

        ϕ_final_mean = get_ϕ_mean_final(prior, eksobj)
        ϕ_init_mean = get_ϕ_mean(prior, eksobj, 1)

        # ϕ_final_mean is transformed mean, not mean transformed parameter
        @test ϕ_final_mean == transform_unconstrained_to_constrained(prior, eks_final_result)
        @test ϕ_final_mean != vec(mean(get_ϕ_final(prior, eksobj), dims = 2))
        # ϕ_init_mean is transformed mean, not mean transformed parameter
        @test ϕ_init_mean == transform_unconstrained_to_constrained(prior, initial_guess)
        @test ϕ_init_mean != vec(mean(get_ϕ(prior, eksobj, 1), dims = 2))
        @test ϕ_init_mean != vec(mean(get_ϕ(prior, eksobj)[1], dims = 2))
        # Regression test of algorithmic efficacy
        @test norm(y_obs .- G(eks_final_result))^2 < norm(y_obs .- G(initial_guess))^2
        @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)

        # Store for comparison with other algorithms
        push!(eks_final_results, eks_final_result)
        push!(eksobjs, eksobj)

    end


end

@testset "EnsembleKalmanInversion" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    iters_with_failure = [5, 8, 9, 15]
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    ekiobj = nothing
    eki_final_result = nothing

    for ((i_prob, inv_problem), loc_method, eks_final_result, eksobj) in
        zip(enumerate(inv_problems), loc_methods, eks_final_results, eksobjs)

        # Get inverse problem
        y_obs, G, Γy, A = inv_problem
        if i_prob == 1
            scheduler = DataMisfitController(on_terminate = "continue")
        else
            scheduler = DefaultScheduler()
        end

        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            localization_method = loc_method,
            scheduler = scheduler,
        )
        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            localization_method = loc_method,
            scheduler = scheduler,
        )

        g_ens = G(get_ϕ_final(prior, ekiobj))
        g_ens_t = permutedims(g_ens, (2, 1))

        @test size(g_ens) == (n_obs, N_ens)

        # EKI iterations
        u_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        for i in 1:N_iter
            # Check SampleSuccGauss handler
            params_i = get_ϕ_final(prior, ekiobj)
            push!(u_i_vec, get_u_final(ekiobj))
            g_ens = G(params_i)
            # Add random failures
            if i in iters_with_failure
                g_ens[:, 1] .= NaN
            end

            EKP.update_ensemble!(ekiobj, g_ens)
            push!(g_ens_vec, g_ens)
            if i == 1
                g_ens_t = permutedims(g_ens, (2, 1))
                @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
            end
            # Correct handling of failures
            @test !any(isnan.(params_i))

            # Check IgnoreFailures handler
            if i <= iters_with_failure[1]
                params_i_unsafe = get_ϕ_final(prior, ekiobj_unsafe)
                g_ens_unsafe = G(params_i_unsafe)
                if i < iters_with_failure[1]
                    EKP.update_ensemble!(ekiobj_unsafe, g_ens_unsafe)
                elseif i == iters_with_failure[1]
                    g_ens_unsafe[:, 1] .= NaN
                    #inconsistent behaviour before/after v1.9 regarding NaNs in matrices
                    if (VERSION.major >= 1) && (VERSION.minor >= 9)
                        # new versions the NaNs break LinearAlgebra.jl
                        @test_throws ArgumentError EKP.update_ensemble!(ekiobj_unsafe, g_ens_unsafe)
                    else
                        # old versions the NaNs pass through LinearAlgebra.jl
                        EKP.update_ensemble!(ekiobj_unsafe, g_ens_unsafe)
                        u_unsafe = get_u_final(ekiobj_unsafe)
                        # Propagation of unhandled failures
                        @test any(isnan.(u_unsafe))
                    end
                end
            end
        end
        push!(u_i_vec, get_u_final(ekiobj))

        @test get_u_prior(ekiobj) == u_i_vec[1]
        @test get_u(ekiobj) == u_i_vec
        @test isequal(get_g(ekiobj), g_ens_vec)
        @test isequal(get_g_final(ekiobj), g_ens_vec[end])
        @test isequal(get_error(ekiobj), ekiobj.err)

        # EKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_final_result = get_u_mean_final(ekiobj)

        g_mean_init = get_g_mean(ekiobj, 1)
        g_mean_final = get_g_mean_final(ekiobj)

        @test eki_init_result == get_u_mean(ekiobj, 1)
        @test eki_final_result == vec(mean(get_u_final(ekiobj), dims = 2))

        ϕ_final_mean = get_ϕ_mean_final(prior, ekiobj)
        ϕ_init_mean = get_ϕ_mean(prior, ekiobj, 1)
        if isa(loc_method, NoLocalization)
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2
        end

        if i_prob <= n_lin_inv_probs && loc_method == NoLocalization()

            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            # EKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - ϕ_final_mean) < norm(ols_mean - ϕ_init_mean)
            # EKS provides a solution closer to the posterior mean -- NOT ROBUST
            # @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - eki_final_result)

            ##### I expect this test to make sense:
            # In words: the ensemble covariance is still a bit ill-dispersed since the
            # algorithm employed still does not include the correction term for finite-sized
            # ensembles.
            @test abs(sum(diag(posterior_cov_inv \ get_u_cov_final(eksobj))) - n_par) > 1e-5
        end

        # Plot evolution of the EKI particles
        if TEST_PLOT_OUTPUT
            gr()
            ϕ_prior = transform_unconstrained_to_constrained(prior, get_u_prior(ekiobj))
            ϕ_final = get_ϕ_final(prior, ekiobj)
            p = plot(ϕ_prior[1, :], ϕ_prior[2, :], seriestype = :scatter, label = "Initial ensemble")
            plot!(ϕ_final[1, :], ϕ_final[2, :], seriestype = :scatter, label = "Final ensemble")
            plot!(
                [ϕ_star[1]],
                xaxis = "cons_p",
                yaxis = "uncons_p",
                seriestype = "vline",
                linestyle = :dash,
                linecolor = :red,
                label = :none,
            )
            plot!([ϕ_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = :none)
            savefig(p, joinpath(@__DIR__, "EKI_test_$(i_prob).png"))
        end
    end
end


@testset "UnscentedKalmanInversion" begin
    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    α_reg = 1.0
    update_freq = 0
    process = Unscented(prior; α_reg = α_reg, update_freq = update_freq, sigma_points = "symmetric")
    iters_with_failure = [5, 8, 9, 15]
    failed_particle_index = [1, 2, 3, 1]

    y_obs, G, Γy, A = inv_problems[n_lin_inv_probs] # lin problem with diag noise

    ukiobj = EKP.EnsembleKalmanProcess(y_obs, Γy, process; rng = rng, failure_handler_method = SampleSuccGauss())
    ukiobj_unsafe = EKP.EnsembleKalmanProcess(y_obs, Γy, process; rng = rng, failure_handler_method = IgnoreFailures())
    # test simplex sigma points
    process_simplex = Unscented(prior; α_reg = α_reg, update_freq = update_freq, sigma_points = "simplex")
    ukiobj_simplex =
        EKP.EnsembleKalmanProcess(y_obs, Γy, process_simplex; rng = rng, failure_handler_method = SampleSuccGauss())

    # Test incorrect construction throws error
    @test_throws ArgumentError Unscented(prior; α_reg = α_reg, update_freq = update_freq, sigma_points = "unknowns")

    # UKI iterations
    u_i_vec = Array{Float64, 2}[]
    g_ens_vec = Array{Float64, 2}[]
    failed_index = 1
    for i in 1:N_iter
        # Check SampleSuccGauss handler
        params_i = get_ϕ_final(prior, ukiobj)
        push!(u_i_vec, get_u_final(ukiobj))
        g_ens = G(params_i)
        # Add random failures
        if i in iters_with_failure
            g_ens[:, failed_particle_index[failed_index]] .= NaN
            failed_index += 1
        end

        EKP.update_ensemble!(ukiobj, g_ens)
        push!(g_ens_vec, g_ens)
        if i == 1
            g_ens_t = permutedims(g_ens, (2, 1))
            @test_throws DimensionMismatch EKP.update_ensemble!(ukiobj, g_ens_t)
        end
        @test !any(isnan.(params_i))

        # Check IgnoreFailures handler
        if i <= iters_with_failure[1]
            params_i_unsafe = get_ϕ_final(prior, ukiobj_unsafe)
            g_ens_unsafe = G(params_i_unsafe)
            if i < iters_with_failure[1]
                EKP.update_ensemble!(ukiobj_unsafe, g_ens_unsafe)
            elseif i == iters_with_failure[1]
                g_ens_unsafe[:, 1] .= NaN
                #inconsistent behaviour before/after v1.9 regarding NaNs in matrices
                if (VERSION.major >= 1) && (VERSION.minor >= 9)
                    # new versions the NaNs break LinearAlgebra.jl
                    @test_throws ArgumentError EKP.update_ensemble!(ukiobj_unsafe, g_ens_unsafe)
                else
                    # old versions the NaNs pass through LinearAlgebra.jl
                    EKP.update_ensemble!(ukiobj_unsafe, g_ens_unsafe)
                    u_unsafe = get_u_final(ukiobj_unsafe)
                    # Propagation of unhandled failures
                    @test any(isnan.(u_unsafe))
                end


            end
        end

        # Update simplex sigma points
        EKP.update_ensemble!(ukiobj_simplex, G(get_ϕ_final(prior, ukiobj_simplex)))
    end
    push!(u_i_vec, get_u_final(ukiobj))

    @test get_u_prior(ukiobj) == u_i_vec[1]
    @test get_u(ukiobj) == u_i_vec
    @test isequal(get_g(ukiobj), g_ens_vec)
    @test isequal(get_g_final(ukiobj), g_ens_vec[end])
    @test isequal(get_error(ukiobj), ukiobj.err)

    @test isa(construct_mean(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
    @test isa(construct_mean(ukiobj, rand(rng, 5, 2 * n_par + 1)), Vector{Float64})
    @test isa(construct_cov(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
    @test isa(construct_cov(ukiobj, rand(rng, 5, 2 * n_par + 1)), Matrix{Float64})
    @test isposdef(construct_cov(ukiobj, construct_sigma_ensemble(ukiobj.process, [0.0; 0.0], [1.0 0; 0 0])))

    # UKI results: Test if ensemble has collapsed toward the true parameter 
    # values
    uki_init_result = vec(mean(get_u_prior(ukiobj), dims = 2))
    uki_final_result = get_u_mean_final(ukiobj)
    uki_simplex_final_result = get_u_mean_final(ukiobj_simplex)
    ϕ_final_mean = get_ϕ_mean_final(prior, ukiobj)
    ϕ_init_mean = get_ϕ_mean(prior, ukiobj, 1)
    u_cov_final = get_u_cov_final(ukiobj)
    u_cov_init = get_u_cov(ukiobj, 1)

    @test ϕ_init_mean == transform_unconstrained_to_constrained(prior, uki_init_result)
    @test ϕ_final_mean == transform_unconstrained_to_constrained(prior, uki_final_result)

    @test tr(u_cov_final) < tr(u_cov_init)
    @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
    @test norm(ϕ_star - transform_unconstrained_to_constrained(prior, uki_simplex_final_result)) <
          norm(ϕ_star - ϕ_init_mean)
    # end

    if TEST_PLOT_OUTPUT
        gr()
        θ_mean_arr = hcat(ukiobj.process.u_mean...)
        N_θ, N_ens = size(θ_mean_arr)
        θθ_std_arr = zeros(Float64, (N_θ, N_ens))
        for i in 1:(N_ens)
            for j in 1:N_θ
                θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j, j])
            end
        end

        u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
        ites = Array(LinRange(1, N_ens, N_ens))
        p = plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "cons_p")
        plot!(ites, fill(u_star[1], N_ens), linestyle = :dash, linecolor = :grey, label = :none)
        plot!(
            ites,
            grid = false,
            θ_mean_arr[2, :],
            yerror = 3.0 * θθ_std_arr[2, :],
            label = "uncons_p",
            xaxis = "Iterations",
        )
        plot!(ites, fill(u_star[2], N_ens), linestyle = :dash, linecolor = :grey, label = :none)
        savefig(p, joinpath(@__DIR__, "UKI_test.png"))
    end
end


@testset "EnsembleKalmanProcess utils" begin
    # Success/failure splitting
    g = rand(5, 10)
    g[:, 1] .= NaN
    successful_ens, failed_ens = split_indices_by_success(g)
    @test length(failed_ens) == 1
    @test length(successful_ens) == size(g, 2) - length(failed_ens)
    for i in 2:7
        g[:, i] .= NaN
    end
    @test_logs (:warn, r"More than 50% of runs produced NaNs") match_mode = :any split_indices_by_success(g)

    u = rand(10, 4)
    @test_logs (:warn, r"Sample covariance matrix over ensemble is singular.") match_mode = :any sample_empirical_gaussian(
        u,
        2,
    )
    @test_throws PosDefException sample_empirical_gaussian(u, 2, inflation = 0.0)

    # Initial ensemble construction
    rng = Random.MersenneTwister(rng_seed)

    ### sanity check on rng:
    d = Parameterized(Normal(0, 1))
    u = ParameterDistribution(Dict("distribution" => d, "constraint" => no_constraint(), "name" => "test"))
    draw_1 = construct_initial_ensemble(rng, u, 1)
    draw_2 = construct_initial_ensemble(u, 1)
    @test !isapprox(draw_1, draw_2)
end
