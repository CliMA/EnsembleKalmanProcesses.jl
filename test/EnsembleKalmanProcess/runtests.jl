using Distributions
using LinearAlgebra
using Random
using Test
using Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses

# Read inverse problem definitions
include("inverse_problem.jl")

n_obs = 12                  # dimension of synthetic observation from G(u)
ϕ_star = [-1.0, 2.0]        # True parameters in constrained space
n_par = size(ϕ_star, 1)
noise_level = 0.1           # Defining the observation noise level (std)
N_ens = 30                  # number of ensemble members
N_iter = 100

# Test different AbstractMatrix types as covariances
obs_corrmats = [1.0 * I, Matrix(1.0 * I, n_obs, n_obs), Diagonal(Matrix(1.0 * I, n_obs, n_obs))]

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
    (
        nonlinear_inv_problem(
            ϕ_star,
            noise_level,
            n_obs,
            rng,
            obs_corrmat = obs_corrmats[3],
            add_or_mult_noise = "mult",
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
    @test isapprox(norm(y_obs .- A * ϕ_star)^2 - n_obs * noise_level^2, 0; atol = 0.06)
end


@testset "Accelerators" begin
    # Get an inverse problem
    y_obs, G, Γy, _ = inv_problems[end - 2] # additive noise inv problem (deterministic map)
    inv_sqrt_Γy = sqrt(inv(Γy))

    rng = Random.MersenneTwister(rng_seed)
    N_ens_tmp = 5
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens_tmp)

    # build accelerated and non-accelerated processes


    ekiobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), accelerator = NesterovAccelerator())
    eksobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior), accelerator = NesterovAccelerator())
    ekiobj_const =
        EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), accelerator = ConstantNesterovAccelerator())
    eksobj_const = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        y_obs,
        Γy,
        Sampler(prior),
        accelerator = ConstantNesterovAccelerator(),
    )
    ekiobj_firstorder = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        y_obs,
        Γy,
        Inversion(),
        accelerator = FirstOrderNesterovAccelerator(),
    )
    eksobj_firstorder = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        y_obs,
        Γy,
        Sampler(prior),
        accelerator = FirstOrderNesterovAccelerator(),
    )

    ekiobj_noacc = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())
    eksobj_noacc = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior))
    ekiobj_noacc_specified =
        EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), accelerator = DefaultAccelerator())
    eksobj_noacc_specified =
        EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior), accelerator = DefaultAccelerator())

    ## test EKP object's accelerator type is consistent (EKP constructor reassigns object in some cases)
    @test typeof(ekiobj.accelerator) <: NesterovAccelerator
    @test typeof(eksobj.accelerator) <: NesterovAccelerator
    @test typeof(ekiobj_const.accelerator) <: ConstantNesterovAccelerator
    @test typeof(eksobj_const.accelerator) <: ConstantNesterovAccelerator
    @test typeof(ekiobj_firstorder.accelerator) <: FirstOrderNesterovAccelerator
    @test typeof(eksobj_firstorder.accelerator) <: FirstOrderNesterovAccelerator
    @test typeof(ekiobj_noacc.accelerator) <: DefaultAccelerator
    @test typeof(eksobj_noacc.accelerator) <: DefaultAccelerator
    @test typeof(ekiobj_noacc_specified.accelerator) <: DefaultAccelerator
    @test typeof(eksobj_noacc_specified.accelerator) <: DefaultAccelerator

    ## test NesterovAccelerators satisfy desired ICs
    @test ekiobj.accelerator.u_prev == initial_ensemble
    @test ekiobj.accelerator.θ_prev == 1.0
    @test eksobj.accelerator.u_prev == initial_ensemble
    @test eksobj.accelerator.θ_prev == 1.0

    @test ekiobj_const.accelerator.λ ≈ 0.9
    @test ekiobj_const.accelerator.u_prev == initial_ensemble
    @test eksobj_const.accelerator.λ ≈ 0.9
    @test eksobj_const.accelerator.u_prev == initial_ensemble

    @test ekiobj_firstorder.accelerator.r ≈ 3.0
    @test ekiobj_firstorder.accelerator.u_prev == initial_ensemble
    @test eksobj_firstorder.accelerator.r ≈ 3.0
    @test eksobj_firstorder.accelerator.u_prev == initial_ensemble

    ## test method convergence
    # Note: this test only requires that the final ensemble is an improvement on the initial ensemble,
    # NOT that the accelerated processes are more effective than the default, as this is not guaranteed.
    # Specific cost values are printed to give an idea of acceleration.
    processes = [
        repeat([Inversion(), TransformInversion(inv(Γy)), Unscented(prior; impose_prior = true)], 2)...,
        Sampler(prior),
    ]
    schedulers = [
        repeat([DefaultScheduler(0.1)], 3)..., # for constant timestep Nesterov
        repeat([DataMisfitController(terminate_at = 100)], 3)..., # for general Nesterov
        EKSStableScheduler(), # for general Nesterov
    ]
    for (process, scheduler) in zip(processes, schedulers)
        accelerators = [
            DefaultAccelerator(),
            ConstantNesterovAccelerator(0.5),
            FirstOrderNesterovAccelerator(),
            NesterovAccelerator(),
        ]
        N_iters = [20, 20, 20, 20]

        init_means = []
        final_means = []

        for (accelerator, N_iter) in zip(accelerators, N_iters)
            process_copy = deepcopy(process)
            scheduler_copy = deepcopy(scheduler)
            println("Accelerator: ", nameof(typeof(accelerator)), " Process: ", nameof(typeof(process_copy)))
            if !(nameof(typeof(process)) == Symbol(Unscented))
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_ensemble,
                    y_obs,
                    Γy,
                    process_copy,
                    rng = copy(rng),
                    scheduler = scheduler_copy,
                    accelerator = accelerator,
                )
            else
                ekpobj = EKP.EnsembleKalmanProcess(
                    y_obs,
                    Γy,
                    process_copy,
                    rng = copy(rng),
                    scheduler = scheduler_copy,
                    accelerator = accelerator,
                )
            end
            ## test get_accelerator function in EKP
            @test ekpobj.accelerator == get_accelerator(ekpobj)

            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)
                g_ens = G(params_i)
                terminated = EKP.update_ensemble!(ekpobj, g_ens)
                if !isnothing(terminated)
                    break
                end
            end
            push!(init_means, vec(mean(get_u_prior(ekpobj), dims = 2)))
            push!(final_means, vec(mean(get_u_final(ekpobj), dims = 2)))

            cost_initial =
                norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, initial_ensemble))))
            cost_final =
                norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[end]))))
            @info "Convergence:" cost_initial cost_final

            u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
            @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[end])))) <
                  norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[end]))))

        end

    end

end


@testset "LearningRateSchedulers" begin

    # Utility
    X = [2 1; 1.1 2] # correct with symmetrisation
    @test isposdef(posdef_correct(X))
    @test posdef_correct(X) ≈ 0.5 * (X + permutedims(X, (2, 1))) atol = 1e-8
    Y = [0 1; -1 0]
    tol = 1e-8
    @test isposdef(posdef_correct(Y, tol = tol)) # symmetrize and add to diagonal
    @test posdef_correct(Y, tol = tol) ≈ tol * I(2) atol = 1e-8



    # Default
    Δt = 3
    dlrs1 = EKP.DefaultScheduler()
    @test dlrs1.Δt_default == Float64(1)
    dlrs2 = EKP.DefaultScheduler(Δt)
    @test dlrs2.Δt_default == Float64(Δt)
    @test EKP.DefaultScheduler() == EKP.DefaultScheduler()

    #Mutable
    mlrs1 = EKP.MutableScheduler()
    @test mlrs1.Δt_mutable == Float64[1]
    mlrs2 = EKP.MutableScheduler(Δt)
    @test mlrs2.Δt_mutable == Float64[Δt]
    @test EKP.MutableScheduler() == EKP.MutableScheduler()
    # EKSStable 
    ekslrs1 = EKP.EKSStableScheduler()
    @test ekslrs1.numerator == Float64(1)
    @test ekslrs1.nugget == Float64(eps())
    @test EKP.EKSStableScheduler() == EKP.EKSStableScheduler()
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
    @test EKP.DataMisfitController() == EKP.DataMisfitController()

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
        Unscented(prior; impose_prior = true),
        #Sparse inversion tests in test/SparseInversion/runtests.jl
    ]
    T_end = 1 # (this could fail a test if N_iters is not enough to reach T_end)
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
            initial_obs_noise_cov = deepcopy(ekpobj.obs_noise_cov)
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
                # ensure Δt is updated
                @test length(ekpobj.Δt) == i
            end
            push!(init_means, vec(mean(get_u_prior(ekpobj), dims = 2)))
            push!(final_means, vec(mean(get_u_final(ekpobj), dims = 2)))
            # ensure obs_noise_cov matrix remains unchanged
            @test initial_obs_noise_cov == ekpobj.obs_noise_cov

            # this test is fine so long as N_iter is large enough to hit the termination time
            if nameof(typeof(scheduler)) == DataMisfitController
                if (scheduler.terminate_at, scheduler.on_terminate) == (Float64(T_end), "stop")
                    @test sum(ekpobj.Δt) ≈ scheduler.terminate_at
                end
            end
        end
        if nameof(typeof(process)) == Inversion
            for i in 1:length(final_means)
                u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
                inv_sqrt_Γy = sqrt(inv(Γy))
                # @test norm(u_star - final_means[i]) < norm(u_star - init_means[i])
                @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i])))) <
                      norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))

            end
        elseif nameof(typeof(process)) == Unscented
            # we are regularizing by the prior, therefore we must account for this in the metric of success
            u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
            inv_sqrt_Γy = sqrt(inv(Γy))
            # compare stats in unconstrained space
            prior_mean = mean(prior)
            inv_sqrt_prior_cov = sqrt(inv(cov(prior)))
            @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i]))))^2 +
                  norm(inv_sqrt_prior_cov * (final_means[i] .- prior_mean))^2 <
                  norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))^2 +
                  norm(inv_sqrt_prior_cov * (init_means[i] .- prior_mean))^2
        end

    end
end

@testset "EnsembleKalmanSampler" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
    @test size(initial_ensemble) == (n_par, N_ens)

    # Global scope to compare against EKS
    global eks_final_results = []
    global eksobjs = []

    # Test EKS for different inverse problem
    for (i_prob, inv_problem) in enumerate(inv_problems)

        # Get inverse problem
        y_obs, G, Γy, _ = inv_problem

        eksobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior); rng = rng)

        params_0 = get_u_final(eksobj)
        g_ens = G(params_0)

        @test size(g_ens) == (n_obs, N_ens)

        if !(size(g_ens, 1) == size(g_ens, 2))
            g_ens_t = permutedims(g_ens, (2, 1))
            @test_throws DimensionMismatch EKP.update_ensemble!(eksobj, g_ens_t)
        end

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
        eks_init_spread = tr(get_u_cov(eksobj, 1))
        eks_final_spread = tr(get_u_cov_final(eksobj))
        @test eks_final_spread < 2 * eks_init_spread # we wouldn't expect the spread to increase much in any one dimension

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
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, eksobj, joinpath(@__DIR__, "EKS_test_$(i_prob).png"))
        end

    end


end

@testset "EnsembleKalmanInversion" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    iters_with_failure = [5, 8, 9, 15]
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    ekiobj = nothing
    eki_final_result = nothing

    for ((i_prob, inv_problem), eks_final_result, eksobj) in zip(enumerate(inv_problems), eks_final_results, eksobjs)

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
            scheduler = deepcopy(scheduler),
        )
        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            scheduler = deepcopy(scheduler),
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
                if !(size(g_ens, 1) == size(g_ens, 2))
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
                end
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
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))

        g_mean_init = get_g_mean(ekiobj, 1)
        g_mean_final = get_g_mean_final(ekiobj)

        @test eki_init_result == get_u_mean(ekiobj, 1)
        @test eki_final_result == vec(mean(get_u_final(ekiobj), dims = 2))

        @test eki_final_spread < 2 * eki_init_spread # we wouldn't expect the spread to increase much in any one dimension

        ϕ_final_mean = get_ϕ_mean_final(prior, ekiobj)
        ϕ_init_mean = get_ϕ_mean(prior, ekiobj, 1)

        if nameof(typeof(ekiobj.localizer)) == EKP.Localizers.NoLocalization
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2
        end

        if i_prob <= n_lin_inv_probs && nameof(typeof(ekiobj.localizer)) == EKP.Localizers.NoLocalization

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
            plot_inv_problem_ensemble(prior, ekiobj, joinpath(@__DIR__, "EKI_test_$(i_prob).png"))
        end
    end
end


@testset "UnscentedKalmanInversion" begin
    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    #  using augmented system (Tikhonov regularization with Kalman inversion in Chada 
    #  et al 2020 and Huang et al (2022)) to regularize the inverse problem, which also imposes prior 
    #  for posterior estimation.
    #  This should be used when the number of observations is smaller than the number
    #  of parameters (ill-posed inverse problems). 
    impose_priors = [false, false, false, true, true, true]
    update_freqs = [1, 1, 1, 0, 0, 0]
    iters_with_failure = [5, 8, 9, 15]
    failed_particle_index = [1, 2, 3, 1]

    ukiobj = nothing
    uki_final_result = nothing

    # checks for the initial vs prior stats 
    proc_tmp = Unscented([1, 1], [1 0; 0 1]; impose_prior = true)
    @test proc_tmp.uu_cov[1] == [1.0 0.0; 0.0 1.0]
    @test proc_tmp.u_mean[1] == [1.0, 1.0]

    # if different initial and prior mean/cov
    proc_tmp = Unscented(prior; impose_prior = true)
    @test proc_tmp.prior_cov == proc_tmp.uu_cov[1]
    @test proc_tmp.prior_mean == proc_tmp.u_mean[1]


    for (i_prob, inv_problem, impose_prior, update_freq) in
        zip(1:length(inv_problems), inv_problems, impose_priors, update_freqs)

        y_obs, G, Γy, A = inv_problem
        scheduler = DataMisfitController(on_terminate = "continue") #will need to be copied as stores run information inside

        process = Unscented(prior; sigma_points = "symmetric", impose_prior = impose_prior, update_freq = update_freq)
        ukiobj = EKP.EnsembleKalmanProcess(
            y_obs,
            Γy,
            process;
            rng = rng,
            scheduler = deepcopy(scheduler),
            failure_handler_method = SampleSuccGauss(),
        )
        ukiobj_unsafe = EKP.EnsembleKalmanProcess(
            y_obs,
            Γy,
            process;
            rng = rng,
            scheduler = deepcopy(scheduler),
            failure_handler_method = IgnoreFailures(),
        )
        # test simplex sigma points
        process_simplex = Unscented(prior; sigma_points = "simplex", impose_prior = impose_prior)
        ukiobj_simplex = EKP.EnsembleKalmanProcess(
            y_obs,
            Γy,
            process_simplex;
            rng = rng,
            scheduler = deepcopy(scheduler),
            failure_handler_method = SampleSuccGauss(),
        )

        # Test incorrect construction throws error
        @test_throws ArgumentError Unscented(prior; sigma_points = "unknowns", impose_prior = impose_prior)

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
                if !(size(g_ens, 1) == size(g_ens, 2))
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(ukiobj, g_ens_t)
                end
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

        @test tr(u_cov_final) < 2 * tr(u_cov_init)
        @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
        @test norm(ϕ_star - transform_unconstrained_to_constrained(prior, uki_simplex_final_result)) <
              norm(ϕ_star - ϕ_init_mean)
        # end
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, ukiobj, joinpath(@__DIR__, "UKI_test_$(i_prob).png"))
            plot_inv_problem_ensemble(prior, ukiobj_simplex, joinpath(@__DIR__, "UKI_test_simplex_$(i_prob).png"))

        end

    end
end

@testset "EnsembleTransformKalmanInversion" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    ekiobj = nothing
    eki_final_result = nothing
    iters_with_failure = [5, 8, 9, 15]

    for (i_prob, inv_problem) in enumerate(inv_problems)

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
            TransformInversion(inv(Γy));
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
        )

        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            TransformInversion(inv(Γy));
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            scheduler = deepcopy(scheduler),
        )


        g_ens = G(get_ϕ_final(prior, ekiobj))
        g_ens_t = permutedims(g_ens, (2, 1))

        @test size(g_ens) == (n_obs, N_ens)

        # ETKI iterations
        u_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        for i in 1:N_iter
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
                if !(size(g_ens, 1) == size(g_ens, 2))
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
                end
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

        # ETKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_final_result = get_u_mean_final(ekiobj)
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))

        g_mean_init = get_g_mean(ekiobj, 1)
        g_mean_final = get_g_mean_final(ekiobj)

        @test eki_init_result == get_u_mean(ekiobj, 1)
        @test eki_final_result == vec(mean(get_u_final(ekiobj), dims = 2))

        @test eki_final_spread < 2 * eki_init_spread # we wouldn't expect the spread to increase much in any one dimension

        ϕ_final_mean = get_ϕ_mean_final(prior, ekiobj)
        ϕ_init_mean = get_ϕ_mean(prior, ekiobj, 1)

        if nameof(typeof(ekiobj.localizer)) == EKP.Localizers.NoLocalization
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2
        end

        if i_prob <= n_lin_inv_probs && nameof(typeof(ekiobj.localizer)) == EKP.Localizers.NoLocalization

            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            # ETKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - ϕ_final_mean) < norm(ols_mean - ϕ_init_mean)
        end

        # Plot evolution of the ETKI particles
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, ekiobj, joinpath(@__DIR__, "ETKI_test_$(i_prob).png"))
        end
    end

    for (i, n_obs_test) in enumerate([10, 10, 100, 1000, 10000])
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        y_obs_test, G_test, Γ_test, A_test =
            linear_inv_problem(ϕ_star, noise_level, n_obs_test, rng; return_matrix = true)

        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs_test,
            Γ_test,
            TransformInversion(inv(Γ_test));
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
        )
        T = 0.0
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ekiobj)
            g_ens = G_test(params_i)

            dt = @elapsed EKP.update_ensemble!(ekiobj, g_ens)
            T += dt
        end
        # Skip timing of first due to precompilation
        if i >= 2
            @info "ETKI with $n_obs_test observations took $T seconds."
        end
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


    rng = Random.MersenneTwister(rng_seed)

    u = rand(10, 4)
    @test_logs (:warn, r"Sample covariance matrix over ensemble is singular.") match_mode = :any sample_empirical_gaussian(
        u,
        2,
    )

    u2 = rand(rng, 5, 20)
    @test all(
        isapprox.(
            sample_empirical_gaussian(copy(rng), u2, 2),
            sample_empirical_gaussian(copy(rng), u2, 2, inflation = 0.0);
            atol = 1e-8,
        ),
    )

    ### sanity check on rng:
    d = Parameterized(Normal(0, 1))
    u = ParameterDistribution(Dict("distribution" => d, "constraint" => no_constraint(), "name" => "test"))
    draw_1 = construct_initial_ensemble(rng, u, 1)
    draw_2 = construct_initial_ensemble(u, 1)
    @test !isapprox(draw_1, draw_2)
end
