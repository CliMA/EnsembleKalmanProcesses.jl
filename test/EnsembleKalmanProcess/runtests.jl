using Distributions
using LinearAlgebra
using Random
using Test
using Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
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
# infinite-time variants, give an initial != prior
initial_1 = Dict("distribution" => Parameterized(Normal(1.5, 0.1)), "constraint" => bounded(-2, 2), "name" => "cons_p")
initial_2 = Dict("distribution" => Parameterized(Normal(1.0, 1)), "constraint" => no_constraint(), "name" => "uncons_p")
initial_dist = ParameterDistribution([initial_1, initial_2])

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


@testset "NaN imputation" begin

    # handling failures.
    mat = randn(7, 4)
    bad_row_vals = 1.0 .* collect(1:size(mat, 1)) # value to replace if whole row is NaN
    nan_tolerance = 0.5 # threshold fraction of mat to determine bad column
    mat[3:4, :] .= NaN
    mat[2, 3] = NaN
    mat[1, [1, 3]] .= NaN
    mat[5, 2] = NaN
    # mat has 2 NaN rows (3&4)
    # mat has column 3 being a failed particle (4/7>nan_tolerance rows failed)
    # mat[1,1] and mat[5,2] are replaceable
    mat_new = impute_over_nans(mat, nan_tolerance, bad_row_vals, verbose = true)
    # check ignored values
    @test sum((mat - mat_new)[.!isnan.(mat - mat_new)]) == 0
    # check there are no "new" NaNs
    @test sum((.!isnan.(mat)) .* isnan.(mat_new)) == 0
    # check changed NaN values
    @test mat_new[1, 1] == mean([mat[1, 2], mat[1, 4]])
    @test mat_new[5, 2] == mean([mat[5, 1], mat[5, 3], mat[5, 4]]) # includes mat[5,3]
    @test all(mat_new[3, [1, 2, 4]] .== bad_row_vals[3])
    @test all(mat_new[4, [1, 2, 4]] .== bad_row_vals[4])

end


@testset "Accelerators" begin
    # Get an inverse problem
    y_obs, G, Γy, _ = inv_problems[end - 2] # additive noise inv problem (deterministic map)
    inv_sqrt_Γy = sqrt(inv(Γy))

    rng = Random.MersenneTwister(rng_seed)
    N_ens_tmp = 20
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

    ekiobj_noacc =
        EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), accelerator = DefaultAccelerator())
    eksobj_noacc =
        EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior), accelerator = DefaultAccelerator())
    ekiobj_default = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())
    eksobj_default = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior))

    ## test EKP object's accelerator type is consistent (EKP constructor reassigns object in some cases)
    @test typeof(get_accelerator(ekiobj)) <: NesterovAccelerator
    @test typeof(get_accelerator(eksobj)) <: NesterovAccelerator
    @test typeof(get_accelerator(ekiobj_const)) <: ConstantNesterovAccelerator
    @test typeof(get_accelerator(eksobj_const)) <: ConstantNesterovAccelerator
    @test typeof(get_accelerator(ekiobj_firstorder)) <: FirstOrderNesterovAccelerator
    @test typeof(get_accelerator(eksobj_firstorder)) <: FirstOrderNesterovAccelerator
    @test typeof(get_accelerator(ekiobj_noacc)) <: DefaultAccelerator
    @test typeof(get_accelerator(eksobj_noacc)) <: DefaultAccelerator
    @test typeof(get_accelerator(ekiobj_default)) <: NesterovAccelerator
    @test typeof(get_accelerator(eksobj_default)) <: DefaultAccelerator

    ## test NesterovAccelerators satisfy desired ICs
    @test get_accelerator(ekiobj).u_prev == initial_ensemble
    @test get_accelerator(ekiobj).θ_prev == 1.0
    @test get_accelerator(eksobj).u_prev == initial_ensemble
    @test get_accelerator(eksobj).θ_prev == 1.0

    @test get_accelerator(ekiobj_const).λ ≈ 0.9
    @test get_accelerator(ekiobj_const).u_prev == initial_ensemble
    @test get_accelerator(eksobj_const).λ ≈ 0.9
    @test get_accelerator(eksobj_const).u_prev == initial_ensemble

    @test get_accelerator(ekiobj_firstorder).r ≈ 3.0
    @test get_accelerator(ekiobj_firstorder).u_prev == initial_ensemble
    @test get_accelerator(eksobj_firstorder).r ≈ 3.0
    @test get_accelerator(eksobj_firstorder).u_prev == initial_ensemble

    ## test method convergence
    # Note: this test only requires that the final ensemble is an improvement on the initial ensemble,
    # NOT that the accelerated processes are more effective than the default, as this is not guaranteed.
    # Specific cost values are printed to give an idea of acceleration.
    processes = [
        repeat(
            [Inversion(), TransformInversion(), Unscented(prior; impose_prior = true), GaussNewtonInversion(prior)],
            2,
        )...,
        Sampler(prior),
    ]
    schedulers = [
        repeat([DefaultScheduler(0.1)], 4)..., # for constant timestep Nesterov
        repeat([DataMisfitController(terminate_at = 100)], 4)..., # for general Nesterov
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
            if !isa(process, Unscented)
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



    # "Default" (i.e. a constant. not the default option in EKI)
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
    @test ekiobj.scheduler == get_scheduler(ekiobj)
    @test get_scheduler(ekiobj) == DataMisfitController(terminate_at = 1)
    @test get_scheduler(eksobj) == EKSStableScheduler{Float64}(1.0, eps())

    #test
    processes = [
        Inversion(),
        TransformInversion(),
        GaussNewtonInversion(prior),
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
        verboses = [true, repeat([false], 4)...]
        N_iters = 20 * ones(5)
        init_means = []
        final_means = []

        for (scheduler, N_iter, verbose) in zip(schedulers, N_iters, verboses)
            println("Scheduler: ", nameof(typeof(scheduler)))
            if !isa(process, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_ensemble,
                    y_obs,
                    Γy,
                    deepcopy(process),
                    rng = copy(rng),
                    scheduler = scheduler,
                    verbose = verbose,
                )
            else #no initial ensemble for UKI
                ekpobj = EKP.EnsembleKalmanProcess(y_obs, Γy, deepcopy(process), rng = copy(rng), scheduler = scheduler)
            end
            initial_obs_noise_cov = deepcopy(get_obs_noise_cov(ekpobj))
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
                @test length(get_Δt(ekpobj)) == i
            end
            push!(init_means, vec(mean(get_u_prior(ekpobj), dims = 2)))
            push!(final_means, vec(mean(get_u_final(ekpobj), dims = 2)))
            # ensure obs_noise_cov matrix remains unchanged
            @test initial_obs_noise_cov == get_obs_noise_cov(ekpobj)

            # this test is fine so long as N_iter is large enough to hit the termination time
            if isa(scheduler, DataMisfitController)
                if (scheduler.terminate_at, scheduler.on_terminate) == (Float64(T_end), "stop")
                    @test sum(get_Δt(ekpobj)) < scheduler.terminate_at + eps()
                end
            end
        end
        if isa(process, Union{Inversion, TransformInversion, GaussNewtonInversion})
            for i in 1:length(final_means)
                u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
                inv_sqrt_Γy = sqrt(inv(Γy))
                # @test norm(u_star - final_means[i]) < norm(u_star - init_means[i])
                @test norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i])))) <
                      norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))

            end
        elseif isa(process, Unscented)
            for i in 1:length(final_means)
                # we are regularizing by the prior, therefore we must account for this in the metric of success
                u_star = transform_constrained_to_unconstrained(prior, ϕ_star)
                inv_sqrt_Γy = sqrt(inv(Γy))
                # compare stats in unconstrained space
                prior_mean = mean(prior)
                inv_sqrt_prior_cov = sqrt(inv(cov(prior)))
                @test norm(
                    inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, final_means[i]))),
                )^2 + norm(inv_sqrt_prior_cov * (final_means[i] .- prior_mean))^2 <
                      norm(inv_sqrt_Γy * (y_obs .- G(transform_unconstrained_to_constrained(prior, init_means[i]))))^2 +
                      norm(inv_sqrt_prior_cov * (init_means[i] .- prior_mean))^2
            end
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

    #check for small ens
    y_obs_tmp, G_tmp, Γy_tmp, A_tmp = inv_problems[1]
    initial_ensemble_small = EKP.construct_initial_ensemble(rng, prior, 9)
    @test_logs (:warn,) EKP.EnsembleKalmanProcess(initial_ensemble_small, y_obs_tmp, Γy_tmp, Inversion())
    prior_60dims = constrained_gaussian("60dims", 0, 1, -Inf, Inf, repeats = 60)
    initial_ensemble_small = EKP.construct_initial_ensemble(rng, prior_60dims, 99)
    @test_logs (:warn,) EKP.EnsembleKalmanProcess(initial_ensemble_small, y_obs_tmp, Γy_tmp, Inversion())

    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    #
    initial_ensemble_inf = EKP.construct_initial_ensemble(copy(rng), initial_dist, N_ens) # for the _inf object initial != prior

    # test process getters
    process = Inversion()
    @test get_prior_mean(process) == nothing
    @test get_prior_cov(process) == nothing
    @test get_impose_prior(process) == false
    @test get_default_multiplicative_inflation(process) == 0.0
    process_inf = Inversion(prior)
    @test isapprox(get_prior_mean(process_inf), Vector(mean(prior)))
    @test isapprox(get_prior_cov(process_inf), Matrix(cov(prior)))
    @test get_impose_prior(process_inf) == true
    @test get_default_multiplicative_inflation(process_inf) == 1e-3

    ekiobj = nothing
    eki_final_result = nothing

    for ((i_prob, inv_problem), eks_final_result, eksobj) in zip(enumerate(inv_problems), eks_final_results, eksobjs)

        # Get inverse problem
        y_obs, G, Γy, A = inv_problem
        if i_prob == 1
            scheduler = DataMisfitController(on_terminate = "continue")
        else
            scheduler = DefaultScheduler(0.1)
        end
        #remove localizers for now
        localization_method = Localizers.NoLocalization()

        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj2 = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = copy(rng),
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
            nan_tolerance = 0.2,
            nan_row_values = 1.0 * collect(1:length(y_obs)),
        )
        ekiobj_nonoise_update = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_inf = EKP.EnsembleKalmanProcess(
            initial_ensemble_inf,
            y_obs,
            Γy,
            Inversion(prior);
            rng = copy(rng),
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        ## some getters in EKP
        @test get_obs(ekiobj) == y_obs
        @test get_obs_noise_cov(ekiobj) == Γy

        g_ens = G(get_ϕ_final(prior, ekiobj))
        g_ens_t = permutedims(g_ens, (2, 1))

        @test size(g_ens) == (n_obs, N_ens)
        @test get_N_ens(ekiobj) == ekiobj.N_ens
        @test get_rng(ekiobj) == ekiobj.rng
        @test get_failure_handler(ekiobj) == ekiobj.failure_handler
        @test get_Δt(ekiobj) == ekiobj.Δt
        # EKI iterations
        u_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        u_i_inf_vec = Array{Float64, 2}[]
        g_ens_inf_vec = Array{Float64, 2}[]
        for i in 1:N_iter
            # Check SampleSuccGauss handler
            params_i = get_ϕ_final(prior, ekiobj)
            push!(u_i_vec, get_u_final(ekiobj))
            g_ens = G(params_i)
            # Add random failures
            if i in iters_with_failure
                # fail particle 1
                g_ens[:, 1] .= NaN

                # add some redeemable failures
                n_nans = 5
                make_nan = shuffle!(rng, collect(1:N_ens))
                g_ens[1, make_nan[1:n_nans]] .= NaN
                make_nan = shuffle!(rng, collect(1:N_ens))
                g_ens[end, make_nan[1:n_nans]] .= NaN

                # quick getter test
                @test get_nan_tolerance(ekiobj) == 0.1 # default
                @test isnothing(get_nan_row_values(ekiobj)) # default
                @test get_nan_tolerance(ekiobj_unsafe) == 0.2
                @test get_nan_row_values(ekiobj_unsafe) == 1.0 * collect(1:length(y_obs))

            end

            EKP.update_ensemble!(ekiobj, g_ens)
            # fix if added redeemable failues
            imputed_g_ens = impute_over_nans(g_ens, 0.1, y_obs)
            push!(g_ens_vec, imputed_g_ens)

            # repeat with inf-time variant
            params_i_inf = get_ϕ_final(prior, ekiobj_inf)
            g_ens_inf = G(params_i_inf)
            if i in iters_with_failure
                g_ens_inf[:, 1] .= NaN
            end
            EKP.update_ensemble!(ekiobj_inf, g_ens_inf)
            push!(g_ens_inf_vec, g_ens_inf)

            if i == 1
                if !(size(g_ens, 1) == size(g_ens, 2))
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
                end

                # test for additional warning if two columns are equal in
                g_ens_nonunique = copy(g_ens)
                g_ens_nonunique[:, 2] = g_ens_nonunique[:, 3]
                @test_logs (:warn,) update_ensemble!(ekiobj2, g_ens_nonunique)

                # test the deterministic flag on only one iteration for errors
                EKP.update_ensemble!(ekiobj_nonoise_update, g_ens, deterministic_forward_map = false)
                @info "No error with flag deterministic_forward_map = false"

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
        @test isequal(get_error(ekiobj), ekiobj.error)

        # EKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_inf_init_result = vec(mean(get_u_prior(ekiobj_inf), dims = 2))
        eki_final_result = get_u_mean_final(ekiobj)
        eki_inf_final_result = get_u_mean_final(ekiobj_inf)
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_inf_init_spread = tr(get_u_cov(ekiobj_inf, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))
        eki_inf_final_spread = tr(get_u_cov_final(ekiobj_inf))

        g_mean_init = get_g_mean(ekiobj, 1)
        g_mean_final = get_g_mean_final(ekiobj)
        g_mean_inf_init = get_g_mean(ekiobj_inf, 1)
        g_mean_inf_final = get_g_mean_final(ekiobj_inf)

        @test eki_init_result == get_u_mean(ekiobj, 1)
        @test eki_final_result == vec(mean(get_u_final(ekiobj), dims = 2))

        @test eki_final_spread < 2 * eki_init_spread # we wouldn't expect the spread to increase much in any one dimension

        ϕ_final_mean = get_ϕ_mean_final(prior, ekiobj)
        ϕ_init_mean = get_ϕ_mean(prior, ekiobj, 1)
        ϕ_inf_final_mean = get_ϕ_mean_final(prior, ekiobj_inf)
        ϕ_inf_init_mean = get_ϕ_mean(prior, ekiobj_inf, 1)

        if isa(get_localizer(ekiobj), EKP.Localizers.NoLocalization)
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2

            @test norm(ϕ_star - ϕ_inf_final_mean) < norm(ϕ_star - ϕ_inf_init_mean)
            @test norm(y_obs .- G(eki_inf_final_result))^2 < norm(y_obs .- G(eki_inf_init_result))^2
            @test norm(y_obs .- g_mean_inf_final)^2 < norm(y_obs .- g_mean_inf_init)^2
        end

        if i_prob <= n_lin_inv_probs && isa(get_localizer(ekiobj), EKP.Localizers.NoLocalization)

            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            # EKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - ϕ_final_mean) < norm(ols_mean - ϕ_init_mean)
            @test norm(ols_mean - ϕ_inf_final_mean) < norm(ols_mean - ϕ_inf_init_mean)
            # EKS provides a solution closer to the posterior mean -- NOT ROBUST
            # @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - eki_final_result)

            # EKI with inf-solution should converge to posterior mean as T-> inf 
            @test norm(posterior_mean - ϕ_inf_final_mean) < norm(posterior_mean - ϕ_inf_init_mean)

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
            deepcopy(process);
            rng = rng,
            scheduler = deepcopy(scheduler),
            failure_handler_method = SampleSuccGauss(),
        )
        ukiobj_unsafe = EKP.EnsembleKalmanProcess(
            y_obs,
            Γy,
            deepcopy(process);
            rng = rng,
            scheduler = deepcopy(scheduler),
            failure_handler_method = IgnoreFailures(),
        )
        # test simplex sigma points
        process_simplex = Unscented(prior; sigma_points = "simplex", impose_prior = impose_prior)
        ukiobj_simplex = EKP.EnsembleKalmanProcess(
            y_obs,
            Γy,
            deepcopy(process_simplex);
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
        @test isequal(get_error(ukiobj), ukiobj.error)

        @test isa(construct_mean(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
        @test isa(construct_mean(ukiobj, rand(rng, 5, 2 * n_par + 1)), Vector{Float64})
        @test isa(construct_cov(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
        @test isa(construct_cov(ukiobj, rand(rng, 5, 2 * n_par + 1)), Matrix{Float64})
        @test isposdef(construct_cov(ukiobj, construct_sigma_ensemble(get_process(ukiobj), [0.0; 0.0], [1.0 0; 0 0])))

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
    initial_ensemble_inf = EKP.construct_initial_ensemble(copy(rng), initial_dist, N_ens) # doesnt need to be the prior

    ekiobj = nothing
    eki_final_result = nothing
    iters_with_failure = [5, 8, 9, 15]

    # test the process getters
    process = TransformInversion()
    @test get_prior_mean(process) == nothing
    @test get_prior_cov(process) == nothing
    @test get_impose_prior(process) == false
    @test isa(get_buffer(process), AbstractVector)
    @test get_default_multiplicative_inflation(process) == 0.0
    process_inf = TransformInversion(prior)
    @test isapprox(get_prior_mean(process_inf), Vector(mean(prior)))
    @test isapprox(get_prior_cov(process_inf), Matrix(cov(prior)))
    @test get_impose_prior(process_inf) == true
    @test isa(get_buffer(process_inf), AbstractVector)
    @test get_default_multiplicative_inflation(process_inf) == 0.0

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
            TransformInversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
        )

        ekiobj_inf = EKP.EnsembleKalmanProcess(
            initial_ensemble_inf,
            y_obs,
            Γy,
            TransformInversion(prior);
            rng = copy(rng),
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
        )

        ekiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            TransformInversion();
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
        u_i_inf_vec = Array{Float64, 2}[]
        g_ens_inf_vec = Array{Float64, 2}[]
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

            # repeat with inf-time variant
            params_i_inf = get_ϕ_final(prior, ekiobj_inf)
            g_ens_inf = G(params_i_inf)
            if i in iters_with_failure
                g_ens_inf[:, 1] .= NaN
            end
            EKP.update_ensemble!(ekiobj_inf, g_ens_inf)
            push!(g_ens_inf_vec, g_ens_inf)

            # check dimensionality
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
        @test isequal(get_error(ekiobj), ekiobj.error)

        # ETKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_inf_init_result = vec(mean(get_u_prior(ekiobj_inf), dims = 2))
        eki_final_result = get_u_mean_final(ekiobj)
        eki_inf_final_result = get_u_mean_final(ekiobj_inf)
        eki_init_spread = tr(get_u_cov(ekiobj, 1))
        eki_inf_init_spread = tr(get_u_cov(ekiobj_inf, 1))
        eki_final_spread = tr(get_u_cov_final(ekiobj))
        eki_inf_final_spread = tr(get_u_cov_final(ekiobj_inf))

        g_mean_init = get_g_mean(ekiobj, 1)
        g_mean_final = get_g_mean_final(ekiobj)
        g_mean_inf_init = get_g_mean(ekiobj_inf, 1)
        g_mean_inf_final = get_g_mean_final(ekiobj_inf)

        @test eki_init_result == get_u_mean(ekiobj, 1)
        @test eki_final_result == vec(mean(get_u_final(ekiobj), dims = 2))

        @test eki_final_spread < 2 * eki_init_spread # we wouldn't expect the spread to increase much in any one dimension

        ϕ_final_mean = get_ϕ_mean_final(prior, ekiobj)
        ϕ_init_mean = get_ϕ_mean(prior, ekiobj, 1)
        ϕ_inf_final_mean = get_ϕ_mean_final(prior, ekiobj_inf)
        ϕ_inf_init_mean = get_ϕ_mean(prior, ekiobj_inf, 1)

        if isa(get_localizer(ekiobj), EKP.Localizers.NoLocalization)
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2

            @test norm(ϕ_star - ϕ_inf_final_mean) < norm(ϕ_star - ϕ_inf_init_mean)
            @test norm(y_obs .- G(eki_inf_final_result))^2 < norm(y_obs .- G(eki_inf_init_result))^2
            @test norm(y_obs .- g_mean_inf_final)^2 < norm(y_obs .- g_mean_inf_init)^2

        end

        if i_prob <= n_lin_inv_probs && isa(get_localizer(ekiobj), EKP.Localizers.NoLocalization)

            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            # ETKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - ϕ_final_mean) < norm(ols_mean - ϕ_init_mean)
            @test norm(ols_mean - ϕ_inf_final_mean) < norm(ols_mean - ϕ_inf_init_mean)

            # EKTI with inf-solution should converge to posterior mean as T-> inf 
            @test norm(posterior_mean - ϕ_inf_final_mean) < norm(posterior_mean - ϕ_inf_init_mean)

        end

        # Plot evolution of the ETKI particles
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, ekiobj, joinpath(@__DIR__, "ETKI_test_$(i_prob).png"))
        end
    end

    n_iter = 10
    for (i, n_obs_test) in enumerate([10, 100, 1000, 10_000, 100_000, 1_000_000]) # also 1_000_000 works (may terminate early)
        # first i effectively ignored - just for precompile!
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        initial_ensemble_inf = EKP.construct_initial_ensemble(copy(rng), initial_dist, N_ens)

        y_obs_test, G_test, Γ_test, A_test =
            linear_inv_problem(ϕ_star, noise_level, n_obs_test, rng; return_matrix = true)
        Γ_test = Diagonal(0.01 * ones(size(y_obs_test)))


        # test the SVD option with low rank approx of a matrix
        Z_test = 0.01 * randn(rng, (n_obs_test, 5))
        Γ_test_svd = tsvd_cov_from_samples(Z_test)
        observation_svd = Observation(Dict(
            "samples" => y_obs_test,
            "covariances" => Γ_test_svd, # should calc the psuedoinverse with SVD properly
            "names" => "cov_as_svd",
        ))
        # also do some kind of transposed form (nonsensical values) to go into other test branch
        ΓT_test_svd = tsvd_mat(Z_test) # just take rank to be dim for a UniformScaling     
        observation_svdT = Observation(Dict(
            "samples" => y_obs_test,
            "covariances" => ΓT_test_svd, # should calc the psuedoinverse with SVD properly
            "names" => "cov_as_svd_transpose",
        ))

        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs_test,
            Γ_test,
            TransformInversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = DataMisfitController(terminate_at = 1e8), # (least scalable scheduler in output-space)
        )
        ekiobj_inf = EKP.EnsembleKalmanProcess(
            initial_ensemble_inf,
            y_obs_test,
            Γ_test,
            TransformInversion(prior);
            rng = copy(rng),
            failure_handler_method = SampleSuccGauss(),
            scheduler = DataMisfitController(terminate_at = 1e8),
        )

        ekiobj_svd = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation_svd,
            TransformInversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = DataMisfitController(terminate_at = 1e8),
        )

        ekiobj_svdT = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation_svdT,
            TransformInversion();
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = DataMisfitController(terminate_at = 1e8),
        )
        n_final = n_iter
        for ekp in (ekiobj, ekiobj_inf, ekiobj_svd, ekiobj_svdT)
            T = 0.0
            for i in 1:n_iter
                params_i = get_ϕ_final(prior, ekp)
                g_ens = G_test(params_i)
                dt = @elapsed begin
                    terminate = EKP.update_ensemble!(ekp, g_ens)
                end
                if !isnothing(terminate)
                    n_final = i - 1
                    break
                end
                T += dt
            end
            # Skip timing of first due to precompilation
            if i >= 2
                @info "$n_final iterations of ETKI with $n_obs_test observations took $T seconds. \n (avg update: $(T/Float64(n_final)))"
                if T / (n_obs_test * Float64(n_final)) > 5e-6 && n_obs_test > 5_000 # tol back-computed from 1_000_000 computation
                    @error "The ETKI update for $(n_obs_test) observations should take under $(n_obs_test*4e-6) per update, received $(T/Float64(n_final)). Significant slowdowns encountered in ETKI"
                end

            end
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


@testset "GaussNewtonKalmanInversion" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)

    iters_with_failure = [5, 8, 9, 15]
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    gnkiobj = nothing
    gnki_final_result = nothing

    # test the GN object construction
    mp = mean(prior)
    cp = cov(prior)
    @test_throws MethodError GaussNewtonInversion(mp, mp)
    @test_throws MethodError GaussNewtonInversion(cp, cp)

    gni_mean_cov = GaussNewtonInversion(mp, cp)
    gni_prior = GaussNewtonInversion(prior)
    @test gni_mean_cov.prior_mean ≈ mp
    @test gni_mean_cov.prior_cov ≈ cp
    @test gni_prior.prior_mean ≈ mp
    @test gni_prior.prior_cov ≈ cp

    for ((i_prob, inv_problem), eks_final_result, eksobj) in zip(enumerate(inv_problems), eks_final_results, eksobjs)

        # Get inverse problem
        y_obs, G, Γy, A = inv_problem
        if i_prob == 1
            scheduler = DataMisfitController() # if terminated too early can miss out later tests
        else
            scheduler = DefaultScheduler(0.001)
        end
        #remove localizers for now
        localization_method = Localizers.NoLocalization()

        gnkiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            GaussNewtonInversion(prior);
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        gnkiobj_unsafe = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            GaussNewtonInversion(prior);
            rng = rng,
            failure_handler_method = IgnoreFailures(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        gnkiobj_nonoise_update = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            GaussNewtonInversion(prior);
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        ## some getters in EKP
        g_ens = G(get_ϕ_final(prior, gnkiobj))
        g_ens_t = permutedims(g_ens, (2, 1))

        # GNKI iterations
        u_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        terminated = nothing
        for i in 1:N_iter
            # Check SampleSuccGauss handler
            params_i = get_ϕ_final(prior, gnkiobj)
            push!(u_i_vec, get_u_final(gnkiobj))
            g_ens = G(params_i)
            # Add random failures
            if i in iters_with_failure
                g_ens[:, 1] .= NaN
            end
            terminated = EKP.update_ensemble!(gnkiobj, g_ens)
            if !isnothing(terminated)
                break
            end

            push!(g_ens_vec, g_ens)
            if i == 1
                if !(size(g_ens, 1) == size(g_ens, 2))
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(gnkiobj, g_ens_t)
                end

                # test the deterministic flag on only one iteration for errors
                EKP.update_ensemble!(gnkiobj_nonoise_update, g_ens, deterministic_forward_map = false)
                @info "No error with flag deterministic_forward_map = false"

            end
            # Correct handling of failures
            @test !any(isnan.(params_i))

            # Check IgnoreFailures handler
            if i <= iters_with_failure[1]
                params_i_unsafe = get_ϕ_final(prior, gnkiobj_unsafe)
                g_ens_unsafe = G(params_i_unsafe)
                if i < iters_with_failure[1]
                    terminated = EKP.update_ensemble!(gnkiobj_unsafe, g_ens_unsafe)
                elseif i == iters_with_failure[1]
                    g_ens_unsafe[:, 1] .= NaN
                    #inconsistent behaviour before/after v1.9 regarding NaNs in matrices
                    if (VERSION.major >= 1) && (VERSION.minor >= 9)
                        # new versions the NaNs break LinearAlgebra.jl
                        @test_throws ArgumentError EKP.update_ensemble!(gnkiobj_unsafe, g_ens_unsafe)
                    else
                        # old versions the NaNs pass through LinearAlgebra.jl
                        EKP.update_ensemble!(gnkiobj_unsafe, g_ens_unsafe)
                        u_unsafe = get_u_final(gnkiobj_unsafe)
                        # Propagation of unhandled failures
                        @test any(isnan.(u_unsafe))
                    end
                end
            end

        end
        if isnothing(terminated)
            push!(u_i_vec, get_u_final(gnkiobj))
        end # if cancelled early then don't need "final iteration"

        @test get_u_prior(gnkiobj) == u_i_vec[1]
        @test get_u(gnkiobj) == u_i_vec
        @test isequal(get_g(gnkiobj), g_ens_vec) # can deal with NaNs
        @test isequal(get_g_final(gnkiobj), g_ens_vec[end])
        @test isequal(get_error(gnkiobj), gnkiobj.error)

        # GNKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        gnki_init_result = vec(mean(get_u_prior(gnkiobj), dims = 2))
        gnki_final_result = get_u_mean_final(gnkiobj)
        gnki_init_spread = tr(get_u_cov(gnkiobj, 1))
        gnki_final_spread = tr(get_u_cov_final(gnkiobj))

        g_mean_init = get_g_mean(gnkiobj, 1)
        g_mean_final = get_g_mean_final(gnkiobj)

        @test gnki_init_result == get_u_mean(gnkiobj, 1)
        @test gnki_final_result == vec(mean(get_u_final(gnkiobj), dims = 2))

        #@info i_prob
        #@info (get_scheduler(gnkiobj))

        #@test gnki_final_spread < 2 * gnki_init_spread # we wouldn't expect the spread to increase much in any one dimension

        ϕ_final_mean = get_ϕ_mean_final(prior, gnkiobj)
        ϕ_init_mean = get_ϕ_mean(prior, gnkiobj, 1)

        if isa(get_localizer(gnkiobj), EKP.Localizers.NoLocalization)
            @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
            @test norm(y_obs .- G(gnki_final_result))^2 < norm(y_obs .- G(gnki_init_result))^2
            @test norm(y_obs .- g_mean_final)^2 < norm(y_obs .- g_mean_init)^2
        end

        if i_prob <= n_lin_inv_probs && isa(get_localizer(gnkiobj), EKP.Localizers.NoLocalization)

            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            # GNKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - ϕ_final_mean) < norm(ols_mean - ϕ_init_mean)
            # EKS provides a solution closer to the posterior mean -- NOT ROBUST
            # @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - gnki_final_result)

            ##### I expect this test to make sense:
            # In words: the ensemble covariance is still a bit ill-dispersed since the
            # algorithm employed still does not include the correction term for finite-sized
            # ensembles.
            @test abs(sum(diag(posterior_cov_inv \ get_u_cov_final(eksobj))) - n_par) > 1e-5
        end

        # Plot evolution of the GNKI particles
        if TEST_PLOT_OUTPUT
            plot_inv_problem_ensemble(prior, gnkiobj, joinpath(@__DIR__, "GNKI_test_$(i_prob).png"))
        end
    end
end
