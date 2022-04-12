using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses

@testset "EnsembleKalmanProcess" begin

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)

    ### sanity check on rng:
    d = Parameterized(Normal(0, 1))
    u = ParameterDistribution(Dict("distribution" => d, "constraint" => no_constraint(), "name" => "test"))
    draw_1 = construct_initial_ensemble(u, 1)
    draw_2 = construct_initial_ensemble(u, 1)
    @test !isapprox(draw_1, draw_2)

    # Re-seed rng
    rng = Random.MersenneTwister(rng_seed)

    ### Generate data from a linear model: a regression problem with n_par parameters
    ### and 1 observation of G(u) = A \times u, where A : R^n_par -> R^n_obs
    n_obs = 30                  # dimension of synthetic observation from G(u)
    n_par = 2                  # Number of parameteres
    u_star = [-1.0, 2.0]          # True parameters
    noise_level = 0.1             # Defining the observation noise level (std) 
    # Test different AbstractMatrices
    Γy_vec =
        [noise_level^2 * I, noise_level^2 * Matrix(I, n_obs, n_obs), noise_level^2 * Diagonal(Matrix(I, n_obs, n_obs))]
    # Test different localizers
    loc_methods = [RBF(2.0), Delta(), NoLocalization()]

    noise = MvNormal(zeros(n_obs), Γy_vec[1])
    C = [1 -.9; -.9 1]          # Correlation structure for linear operator
    A = rand(rng, MvNormal(zeros(2,), C), n_obs)'    # Linear operator in R^{n_par x n_obs}

    @test size(A) == (n_obs, n_par)

    #### Define linear model
    function G(u)
        A * u
    end
    # Define nonlinear model
    function G₁(u)
        [sqrt((u[1] - u_star[1])^2 + (u[2] - u_star[2])^2)]
    end

    y_star = G(u_star)
    y_obs = y_star .+ rand(rng, noise)

    @test size(y_obs) == (n_obs,)


    # sum(y-G)^2 ~ n_obs*noise_level^2
    @test isapprox(norm(y_obs .- G(u_star))^2 - n_obs * noise_level^2, 0; atol = 0.05)

    #### Define prior information on parameters
    prior_u1 = Dict("distribution" => Parameterized(Normal(0.0, 0.5)), "constraint" => no_constraint(), "name" => "u1")
    prior_u2 = Dict("distribution" => Parameterized(Normal(3.0, 0.5)), "constraint" => no_constraint(), "name" => "u2")
    prior = ParameterDistribution([prior_u1, prior_u2])

    prior_mean = mean(prior)

    # Assuming independence of u1 and u2
    prior_cov = cov(prior) #convert(Array, Diagonal([sqrt(2.), sqrt(2.)]))


    @testset "EnsembleKalmanSampler" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        N_ens = 50 # number of ensemble members (NB for @test throws, make different to N_ens)
        N_iter = 20

        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        @test size(initial_ensemble) == (n_par, N_ens)

        # Global scope to compare against EKI
        global eks_final_result = nothing
        global eksobj = nothing
        for Γy in Γy_vec
            eksobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior_mean, prior_cov); rng = rng)

            g_ens = G(get_u_final(eksobj))
            @test size(g_ens) == (n_obs, N_ens)
            # as the columns of g are the data, this should throw an error

            # EKS iterations
            for i in 1:N_iter
                params_i = get_u_final(eksobj)
                g_ens = G(params_i)
                if i == 1
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(eksobj, g_ens_t)
                end

                EKP.update_ensemble!(eksobj, g_ens)
            end

            eks_final_result = vec(mean(get_u_final(eksobj), dims = 2))
        end

        # Plot evolution of the EKS particles
        if TEST_PLOT_OUTPUT
            gr()
            p = plot(
                get_u_prior(eksobj)[1, :],
                get_u_prior(eksobj)[2, :],
                seriestype = :scatter,
                label = "Initial ensemble",
            )
            plot!(get_u_final(eksobj)[1, :], get_u_final(eksobj)[2, :], seriestype = :scatter, label = "Final ensemble")
            plot!(
                [u_star[1]],
                xaxis = "u1",
                yaxis = "u2",
                seriestype = "vline",
                linestyle = :dash,
                linecolor = :red,
                label = :none,
            )
            plot!([u_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = :none)
            savefig(p, "EKS_test.png")
        end
    end


    @testset "EnsembleKalmanInversion" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        N_ens = 50 # number of ensemble members (NB for @test throws, make different to N_ens)
        N_iter = 20
        iters_with_failure = [5, 8, 9, 15]
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        ekiobj = nothing
        eki_final_result = nothing
        for (Γy, loc_method) in zip(Γy_vec, loc_methods)
            ekiobj = EKP.EnsembleKalmanProcess(
                initial_ensemble,
                y_obs,
                Γy,
                Inversion();
                rng = rng,
                failure_handler_method = SampleSuccGauss(),
                localization_method = loc_method,
            )
            ekiobj_unsafe = EKP.EnsembleKalmanProcess(
                initial_ensemble,
                y_obs,
                Γy,
                Inversion();
                rng = rng,
                failure_handler_method = IgnoreFailures(),
                localization_method = loc_method,
            )

            # some checks 
            g_ens = G(get_u_final(ekiobj))
            @test size(g_ens) == (n_obs, N_ens)
            # as the columns of g are the data, this should throw an error
            g_ens_t = permutedims(g_ens, (2, 1))
            @test_throws DimensionMismatch find_ekp_stepsize(ekiobj, g_ens_t)
            Δ = find_ekp_stepsize(ekiobj, g_ens)
            # huge collapse for linear problem so should find timestep should < 1
            if isa(loc_method, NoLocalization)
                @test Δ < 1
            end
            # NOTE We don't use this info, this is just for the test.

            # EKI iterations
            params_i_vec = []
            g_ens_vec = []
            for i in 1:N_iter
                # Check SampleSuccGauss handler
                params_i = get_u_final(ekiobj)
                push!(params_i_vec, params_i)
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
                @test !any(isnan.(params_i))

                # Check IgnoreFailures handler
                if i <= iters_with_failure[1]
                    params_i_unsafe = get_u_final(ekiobj_unsafe)
                    g_ens_unsafe = G(params_i_unsafe)
                    if i < iters_with_failure[1]
                        EKP.update_ensemble!(ekiobj_unsafe, g_ens_unsafe)
                    elseif i == iters_with_failure[1]
                        g_ens_unsafe[:, 1] .= NaN
                        EKP.update_ensemble!(ekiobj_unsafe, g_ens_unsafe)
                        u_unsafe = get_u_final(ekiobj_unsafe)
                        @test any(isnan.(u_unsafe))
                    end
                end
            end
            push!(params_i_vec, get_u_final(ekiobj))

            @test get_u_prior(ekiobj) == params_i_vec[1]
            @test get_u(ekiobj) == params_i_vec
            @test isequal(get_g(ekiobj), g_ens_vec)
            @test isequal(get_g_final(ekiobj), g_ens_vec[end])
            @test isequal(get_error(ekiobj), ekiobj.err)

            # EKI results: Test if ensemble has collapsed toward the true parameter 
            # values
            eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
            eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
            if isa(loc_method, NoLocalization)
                @test norm(u_star - eki_final_result) < norm(u_star - eki_init_result)
            end
        end

        # Plot evolution of the EKI particles
        #eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
        if TEST_PLOT_OUTPUT
            gr()
            p = plot(
                get_u_prior(ekiobj)[1, :],
                get_u_prior(ekiobj)[2, :],
                seriestype = :scatter,
                label = "Initial ensemble",
            )
            plot!(get_u_final(ekiobj)[1, :], get_u_final(ekiobj)[2, :], seriestype = :scatter, label = "Final ensemble")
            plot!(
                [u_star[1]],
                xaxis = "u1",
                yaxis = "u2",
                seriestype = "vline",
                linestyle = :dash,
                linecolor = :red,
                label = :none,
            )
            plot!([u_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = :none)
            savefig(p, "EKI_test.png")
        end

        for Γy in Γy_vec
            posterior_cov_inv = (A' * (Γy \ A) + 1 * Matrix(I, n_par, n_par) / prior_cov)
            ols_mean = (A' * (Γy \ A)) \ (A' * (Γy \ y_obs))
            posterior_mean = posterior_cov_inv \ ((A' * (Γy \ A)) * ols_mean + (prior_cov \ prior_mean))

            #### This tests correspond to:
            # NOTE THESE ARE VERY SENSITIVE TO THE CORRESPONDING N_iter
            # EKI provides a solution closer to the ordinary Least Squares estimate
            @test norm(ols_mean - eki_final_result) < norm(ols_mean - eks_final_result)
            # EKS provides a solution closer to the posterior mean
            @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - eki_final_result)

            ##### I expect this test to make sense:
            # In words: the ensemble covariance is still a bit ill-dispersed since the
            # algorithm employed still does not include the correction term for finite-sized
            # ensembles.
            @test abs(sum(diag(posterior_cov_inv \ cov(get_u_final(eksobj), dims = 2))) - n_par) > 1e-5
        end
    end


    @testset "UnscentedKalmanInversion" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        N_iter = 20 # number of UKI iterations
        α_reg = 1.0
        update_freq = 0
        process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
        iters_with_failure = [5, 8, 9, 15]
        failed_particle_index = [1, 2, 3, 1]
        Γy = Γy_vec[3]
        ukiobj = EKP.EnsembleKalmanProcess(y_star, Γy, process; rng = rng, failure_handler_method = SampleSuccGauss())
        ukiobj_unsafe =
            EKP.EnsembleKalmanProcess(y_star, Γy, process; rng = rng, failure_handler_method = IgnoreFailures())
        # UKI iterations
        params_i_vec = []
        g_ens_vec = []
        failed_index = 1
        for i in 1:N_iter
            # Check SampleSuccGauss handler
            params_i = get_u_final(ukiobj)
            push!(params_i_vec, params_i)
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
                params_i_unsafe = get_u_final(ukiobj_unsafe)
                g_ens_unsafe = G(params_i_unsafe)
                if i < iters_with_failure[1]
                    EKP.update_ensemble!(ukiobj_unsafe, g_ens_unsafe)
                elseif i == iters_with_failure[1]
                    g_ens_unsafe[:, 1] .= NaN
                    EKP.update_ensemble!(ukiobj_unsafe, g_ens_unsafe)
                    u_unsafe = get_u_final(ukiobj_unsafe)
                    @test any(isnan.(u_unsafe))
                end
            end

        end
        push!(params_i_vec, get_u_final(ukiobj))

        @test get_u_prior(ukiobj) == params_i_vec[1]
        @test get_u(ukiobj) == params_i_vec
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
        @test norm(u_star - uki_final_result) < norm(u_star - uki_init_result)
        # end

        if TEST_PLOT_OUTPUT
            gr()
            θ_mean_arr = hcat(ukiobj.process.u_mean...)
            N_θ = length(ukiobj.process.u_mean[1])
            θθ_std_arr = zeros(Float64, (N_θ, N_iter + 1))
            for i in 1:(N_iter + 1)
                for j in 1:N_θ
                    θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j, j])
                end
            end

            ites = Array(LinRange(1, N_iter + 1, N_iter + 1))
            p = plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "u1")
            plot!(ites, fill(u_star[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = :none)
            plot!(
                ites,
                grid = false,
                θ_mean_arr[2, :],
                yerror = 3.0 * θθ_std_arr[2, :],
                label = "u2",
                xaxis = "Iterations",
            )
            plot!(ites, fill(u_star[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = :none)
            savefig(p, "UKI_test.png")
        end

        ### Generate data from G₁(u) with a sparse u
        n_obs = 1                  # Number of synthetic observations from G₁(u)
        n_par = 2                  # Number of parameteres
        u_star = [1.0, 0.0]          # True parameters
        noise_level = 1e-3            # Defining the observation noise level
        Γy = noise_level * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations
        noise = MvNormal(zeros(n_obs), Γy)

        y_star = G₁(u_star)
        y_obs = y_star + rand(rng, noise)

        @test size(y_star) == (n_obs,)

        #### Define prior information on parameters
        prior_u1 =
            Dict("distribution" => Parameterized(Normal(0.0, 2)), "constraint" => no_constraint(), "name" => "u1")
        prior_u2 =
            Dict("distribution" => Parameterized(Normal(3.0, 2)), "constraint" => no_constraint(), "name" => "u2")
        prior = ParameterDistribution([prior_u1, prior_u2])
    end

    ###
    ###  Calibrate (4): Sparse Ensemble Kalman Inversion
    ###
    @testset "SparseInversion" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        n_obs = 1
        n_par = 2
        N_ens = 20 # number of ensemble members
        N_iter = 5 # number of EKI iterations
        iters_with_failure = [1, 3]
        Γy_vec = [noise_level * Matrix(I, n_obs, n_obs), noise_level * I]
        loc_methods = [NoLocalization(), RBF(2.0)]
        # Sparse EKI parameters
        γ = 1.0
        regs = [1e-4, 1e-3]
        uc_idxs = [[1, 2], :]

        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        @test size(initial_ensemble) == (n_par, N_ens)

        threshold_values = [0, 1e-2]
        test_names = ["test", "test_thresholded"]

        for (threshold_value, reg, uc_idx, test_name, Γy, loc_method) in
            zip(threshold_values, regs, uc_idxs, test_names, Γy_vec, loc_methods)
            process = SparseInversion(γ, threshold_value, uc_idx, reg)

            ekiobj = EKP.EnsembleKalmanProcess(
                initial_ensemble,
                y_obs,
                Γy,
                process;
                rng = rng,
                failure_handler_method = SampleSuccGauss(),
                localization_method = loc_method,
            )
            ekiobj_unsafe = EKP.EnsembleKalmanProcess(
                initial_ensemble,
                y_obs,
                Γy,
                process;
                rng = rng,
                failure_handler_method = IgnoreFailures(),
            )

            # EKI iterations
            params_i_vec = []
            g_ens_vec = []
            for i in 1:N_iter
                # Check SammpleSuccGauss handler
                params_i = get_u_final(ekiobj)
                push!(params_i_vec, params_i)
                g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ens]...)
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
                    EKP.update_ensemble!(ekiobj, g_ens, Δt_new = ekiobj.Δt[1])
                end
                @test !any(isnan.(params_i))
            end
            push!(params_i_vec, get_u_final(ekiobj))

            @test get_u_prior(ekiobj) == params_i_vec[1]
            @test get_u(ekiobj) == params_i_vec
            @test isequal(get_g(ekiobj), g_ens_vec)
            @test isequal(get_g_final(ekiobj), g_ens_vec[end])
            @test isequal(get_error(ekiobj), ekiobj.err)

            # EKI results: Test if ensemble has collapsed toward the true parameter
            # values
            eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
            eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
            @test norm(u_star - eki_final_result) < norm(u_star - eki_init_result)
            @test sum(eki_final_result .> 0.05) < size(eki_final_result)[1]

            # Plot evolution of the EKI particles
            eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))

            if TEST_PLOT_OUTPUT
                gr()
                p = plot(get_u_prior(ekiobj)[1, :], get_u_prior(ekiobj)[2, :], seriestype = :scatter)
                plot!(get_u_final(ekiobj)[1, :], get_u_final(ekiobj)[2, :], seriestype = :scatter)
                plot!(
                    [u_star[1]],
                    xaxis = "u1",
                    yaxis = "u2",
                    seriestype = "vline",
                    linestyle = :dash,
                    linecolor = :red,
                )
                plot!([u_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red)
                savefig(p, string("SparseEKI_", test_name, ".png"))
            end

            # Test other constructors
            @test isa(SparseInversion(γ), SparseInversion)

        end
    end

    @testset "EnsembleKalmanProcess utils" begin
        g = rand(5, 10)
        g[:, 1] .= NaN
        successful_ens, failed_ens = split_indices_by_success(g)
        @test length(failed_ens) == 1
        @test length(successful_ens) == size(g, 2) - length(failed_ens)
        for i in 2:7
            g[:, i] .= NaN
        end
        @test_logs (:warn,) split_indices_by_success(g)

        u = rand(10, 4)
        @test_logs (:warn,) sample_empirical_gaussian(u, 2)
        @test_throws PosDefException sample_empirical_gaussian(u, 2, inflation = 0.0)
    end

end
