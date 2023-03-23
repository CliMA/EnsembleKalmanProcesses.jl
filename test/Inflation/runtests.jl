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
include("../EnsembleKalmanProcess/inverse_problem.jl")

n_obs = 30                  # dimension of synthetic observation from G(u)
ϕ_star = [-1.0, 2.0]        # True parameters in constrained space
n_par = size(ϕ_star, 1)
noise_level = 0.1           # Defining the observation noise level (std)
N_ens = 1000               # number of ensemble members
N_iter = 1                  # number of EKI iterations

obs_corrmat = Diagonal(Matrix(I, n_obs, n_obs))

prior_1 = Dict("distribution" => Parameterized(Normal(0.0, 0.5)), "constraint" => bounded(-2, 2), "name" => "cons_p")
prior_2 = Dict("distribution" => Parameterized(Normal(3.0, 0.5)), "constraint" => no_constraint(), "name" => "uncons_p")
prior = ParameterDistribution([prior_1, prior_2])
prior_cov = cov(prior)

rng_seed = 42
rng = Random.MersenneTwister(rng_seed)

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

Δts = [0.5, 0.75]

@testset "Inflation" begin

    ekiobj = nothing

    for Δt_i in 1:length(Δts)
        Δt = Δts[Δt_i]

        # Get inverse problem
        y_obs, G, Γy, A =
            linear_inv_problem(ϕ_star, noise_level, n_obs, prior, rng; obs_corrmat = obs_corrmat, return_matrix = true)

        ekiobj = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            Γy,
            Inversion();
            Δt = Δt,
            rng = rng,
            failure_handler_method = SampleSuccGauss(),
        )

        g_ens = G(get_u_final(ekiobj))

        # ensure error is thrown when scaled time step >= 1
        @test_throws ErrorException EKP.update_ensemble!(ekiobj, g_ens; multiplicative_inflation = true, s = 3.0)
        @test_throws ErrorException EKP.update_ensemble!(ekiobj, g_ens; additive_inflation = true, s = 3.0)

        # EKI iterations
        for i in 1:N_iter
            # Check SampleSuccGauss handler
            params_i = get_u_final(ekiobj)

            g_ens = G(params_i)

            # standard update
            EKP.update_ensemble!(ekiobj, g_ens, EKP.get_process(ekiobj))
            eki_mult_inflation = deepcopy(ekiobj)
            eki_add_inflation = deepcopy(ekiobj)
            eki_add_inflation_prior = deepcopy(ekiobj)

            # multiplicative inflation after standard update
            EKP.multiplicative_inflation!(eki_mult_inflation)
            # additive inflation after standard update
            EKP.additive_inflation!(eki_add_inflation)
            # additive inflation (scaling prior cov) after standard update
            EKP.additive_inflation!(eki_add_inflation_prior; use_prior_cov = true)

            # ensure multiplicative inflation approximately preserves ensemble mean
            @test get_u_mean_final(ekiobj) ≈ get_u_mean_final(eki_mult_inflation) atol = 0.2
            # ensure additive inflation approximately preserves ensemble mean
            @test get_u_mean_final(ekiobj) ≈ get_u_mean_final(eki_add_inflation) atol = 0.2
            # ensure additive inflation (scaling prior cov) approximately preserves ensemble mean
            @test get_u_mean_final(ekiobj) ≈ get_u_mean_final(eki_add_inflation_prior) atol = 0.2

            # ensure inflation expands ensemble variance as expected
            expected_var_gain = 1 / (1 - Δt)
            @test get_u_cov_final(ekiobj) .* expected_var_gain ≈ get_u_cov_final(eki_mult_inflation) atol = 1e-3
            # implemented default additive, multiplicative inflation have same effect on ensemble covariance
            @test get_u_cov_final(eki_add_inflation) ≈ get_u_cov_final(eki_mult_inflation) atol = 1e-3
            # ensure additive inflation with prior affects variance as expected
            # note: we accept a higher relative tolerance here because the 2 parameter ensemble collapses
            # note: quickly so the added noise (scaled from prior) is relatively large (difference eliminated with large ensemble)
            @test get_u_cov_final(eki_add_inflation_prior) - get_u_cov_final(ekiobj) ≈
                  (Δt * expected_var_gain) .* prior_cov atol = 0.2

            # ensure inflation is only added in final iteration
            u_standard = get_u(ekiobj)
            u_mult_inflation = get_u(eki_mult_inflation)
            u_add_inflation = get_u(eki_add_inflation)
            @test u_standard[1:(end - 1)] == u_mult_inflation[1:(end - 1)]
            @test u_standard[1:(end - 1)] == u_add_inflation[1:(end - 1)]
            @test u_standard[end] != u_mult_inflation[end]
            @test u_standard[end] != u_add_inflation[end]

        end
    end
    # inflation update should not affect initial parameter ensemble (drawn from prior)
    @test get_u_prior(ekiobj) == initial_ensemble
end
