using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses: construct_mean, construct_cov
const EKP = EnsembleKalmanProcesses

@testset "EnsembleKalmanProcess" begin

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)

    ### Generate data from a linear model: a regression problem with n_par parameters
    ### and 1 observation of G(u) = A \times u, where A : R^n_par -> R^n_obs
    n_obs = 10                  # dimension of synthetic observation from G(u)
    n_par = 2                  # Number of parameteres
    u_star = [-1.0, 2.0]          # True parameters
    noise_level = 0.05            # Defining the observation noise level (std) 
    Γy = noise_level^2 * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations
    noise = MvNormal(zeros(n_obs), Γy)
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
    prior_distns = [Parameterized(Normal(0.0, 0.5)), Parameterized(Normal(3.0, 0.5))]
    constraints = [[no_constraint()], [no_constraint()]]
    prior_names = ["u1", "u2"]
    prior = ParameterDistribution(prior_distns, constraints, prior_names)

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

        global eksobj =
            EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Sampler(prior_mean, prior_cov); rng = rng)

        g_ens = G(get_u_final(eksobj))
        @test size(g_ens) == (n_obs, N_ens)
        # as the columns of g are the data, this should throw an error
        g_ens_t = permutedims(g_ens, (2, 1))

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
        # Plot evolution of the EKS particles
        global eks_final_result = vec(mean(get_u_final(eksobj), dims = 2))

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

        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        ekiobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(); rng = rng)

        # some checks 
        g_ens = G(get_u_final(ekiobj))
        @test size(g_ens) == (n_obs, N_ens)
        # as the columns of g are the data, this should throw an error
        g_ens_t = permutedims(g_ens, (2, 1))
        @test_throws DimensionMismatch find_ekp_stepsize(ekiobj, g_ens_t)
        Δ = find_ekp_stepsize(ekiobj, g_ens)
        # huge collapse for linear problem so should find timestep should < 1
        @test Δ < 1
        # NOTE We don't use this info, this is just for the test.

        # EKI iterations
        params_i_vec = []
        g_ens_vec = []
        for i in 1:N_iter
            params_i = get_u_final(ekiobj)
            push!(params_i_vec, params_i)
            g_ens = G(params_i)
            push!(g_ens_vec, g_ens)
            if i == 1
                g_ens_t = permutedims(g_ens, (2, 1))
                @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
            end
            EKP.update_ensemble!(ekiobj, g_ens)

        end
        push!(params_i_vec, get_u_final(ekiobj))

        @test get_u_prior(ekiobj) == params_i_vec[1]
        @test get_u(ekiobj) == params_i_vec
        @test get_g(ekiobj) == g_ens_vec
        @test get_g_final(ekiobj) == g_ens_vec[end]
        @test get_error(ekiobj) == ekiobj.err

        # EKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
        @test norm(u_star - eki_final_result) < norm(u_star - eki_init_result)

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


    @testset "UnscentedKalmanInversion" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        N_iter = 20 # number of UKI iterations
        α_reg = 1.0
        update_freq = 0
        process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
        ukiobj = EKP.EnsembleKalmanProcess(y_star, Γy, process; rng = rng)

        # UKI iterations
        params_i_vec = []
        g_ens_vec = []
        for i in 1:N_iter
            params_i = get_u_final(ukiobj)
            push!(params_i_vec, params_i)
            g_ens = G(params_i)
            push!(g_ens_vec, g_ens)
            if i == 1
                g_ens_t = permutedims(g_ens, (2, 1))
                @test_throws DimensionMismatch EKP.update_ensemble!(ukiobj, g_ens_t)
            end
            EKP.update_ensemble!(ukiobj, g_ens)
        end
        push!(params_i_vec, get_u_final(ukiobj))

        @test get_u_prior(ukiobj) == params_i_vec[1]
        @test get_u(ukiobj) == params_i_vec
        @test get_g(ukiobj) == g_ens_vec
        @test get_g_final(ukiobj) == g_ens_vec[end]
        @test get_error(ukiobj) == ukiobj.err

        @test isa(construct_mean(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
        @test isa(construct_mean(ukiobj, rand(rng, 5, 2 * n_par + 1)), Vector{Float64})
        @test isa(construct_cov(ukiobj, rand(rng, 2 * n_par + 1)), Float64)
        @test isa(construct_cov(ukiobj, rand(rng, 5, 2 * n_par + 1)), Matrix{Float64})

        # UKI results: Test if ensemble has collapsed toward the true parameter 
        # values
        uki_init_result = vec(mean(get_u_prior(ukiobj), dims = 2))
        uki_final_result = get_u_mean_final(ukiobj)
        @test norm(u_star - uki_final_result) < norm(u_star - uki_init_result)

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
        prior_distributions = [Parameterized(Normal(0, 2)), Parameterized(Normal(0, 2))]
        constraints = [[no_constraint()], [no_constraint()]]
        prior_names = ["u1", "u2"]
        prior = ParameterDistribution(prior_distns, constraints, prior_names)
    end

    ###
    ###  Calibrate (4): Sparse Ensemble Kalman Inversion
    ###
    @testset "SparseInversion" begin
        # Seed for pseudo-random number generator
        rng_seed = 42
        rng = Random.MersenneTwister(rng_seed)

        N_ens = 20 # number of ensemble members
        N_iter = 5 # number of EKI iterations
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)
        @test size(initial_ensemble) == (n_par, N_ens)

        for (threshold_eki, threshold_value, test_name) in ((false, 1e-2, "test"), (true, 1e-2, "test_thresholded"))

            # Sparse EKI parameters
            γ = 1.0
            reg = 1e-4
            uc_idx = [1, 2]

            process = SparseInversion(γ, threshold_eki, threshold_value, reg, uc_idx)

            ekiobj = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, process; rng = rng)

            # EKI iterations
            params_i_vec = []
            g_ens_vec = []
            for i in 1:N_iter
                params_i = get_u_final(ekiobj)
                push!(params_i_vec, params_i)
                g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ens]...)
                push!(g_ens_vec, g_ens)
                if i == 1
                    g_ens_t = permutedims(g_ens, (2, 1))
                    @test_throws DimensionMismatch EKP.update_ensemble!(ekiobj, g_ens_t)
                    EKP.update_ensemble!(ekiobj, g_ens)
                else
                    EKP.update_ensemble!(ekiobj, g_ens, Δt_new = ekiobj.Δt[1])
                end
            end
            push!(params_i_vec, get_u_final(ekiobj))

            @test get_u_prior(ekiobj) == params_i_vec[1]
            @test get_u(ekiobj) == params_i_vec
            @test get_g(ekiobj) == g_ens_vec
            @test get_g_final(ekiobj) == g_ens_vec[end]
            @test get_error(ekiobj) == ekiobj.err

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
        end
    end
end
