using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
const EKP = EnsembleKalmanProcesses

TEST_PLOT_OUTPUT = false

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
    obs_corrmats = [I, Matrix(I, n_obs, n_obs), Diagonal(Matrix(I, n_obs, n_obs))]
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
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)
    nl_inv_problems = [
        nonlinear_inv_problem(ϕ_star, noise_level, n_obs, prior, rng; obs_corrmat = corrmat) for corrmat in obs_corrmats
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

    for (threshold_value, reg, uc_idx, test_name, lin_inv_problem, loc_method) in
        zip(threshold_values, regs, uc_idxs, test_names, nl_inv_problems, loc_methods)

        y_obs, G, Γy = lin_inv_problem

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
        params_i_vec = Array{Float64, 2}[]
        g_ens_vec = Array{Float64, 2}[]
        for i in 1:N_iter
            # Check SammpleSuccGauss handler
            params_i = get_u_final(ekiobj)
            push!(params_i_vec, params_i)
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

        # EKI results: Test if ensemble has collapsed toward the true constrained parameter
        # values
        eki_init_result = vec(mean(get_u_prior(ekiobj), dims = 2))
        eki_final_result = vec(mean(get_u_final(ekiobj), dims = 2))
        ϕ_final_mean = transform_unconstrained_to_constrained(prior, eki_final_result)
        ϕ_init_mean = transform_unconstrained_to_constrained(prior, eki_init_result)
        @test norm(ϕ_star - ϕ_final_mean) < norm(ϕ_star - ϕ_init_mean)
        @test norm(y_obs .- G(eki_final_result))^2 < norm(y_obs .- G(eki_init_result))^2

        # Plot evolution of the EKI particles in constrained space
        if TEST_PLOT_OUTPUT
            gr()
            ϕ_prior = transform_unconstrained_to_constrained(prior, get_u_prior(ekiobj))
            ϕ_final = transform_unconstrained_to_constrained(prior, get_u_final(ekiobj))
            p = plot(ϕ_prior[1, :], ϕ_prior[2, :], seriestype = :scatter)
            plot!(ϕ_final[1, :], ϕ_final[2, :], seriestype = :scatter)
            plot!(
                [ϕ_star[1]],
                xaxis = "cons_p",
                yaxis = "uncons_p",
                seriestype = "vline",
                linestyle = :dash,
                linecolor = :red,
            )
            plot!([ϕ_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red)
            savefig(p, string("SparseEKI_", test_name, ".png"))
        end

        # Test other constructors
        @test isa(SparseInversion(γ), SparseInversion)

    end
end
