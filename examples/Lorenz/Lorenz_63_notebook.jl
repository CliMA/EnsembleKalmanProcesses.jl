# %% Imports
include("GModel_L63.jl") # Contains Lorenz 63 source code
include("Lorenz_63_notebook_utils.jl") # Preliminaries / data construction / plotting helpers

using Distributions  # probability distributions and associated functions
using ForwardDiff
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

const EKP = EnsembleKalmanProcesses

gr()

# %% Tunable parameters
# Play with any of these and re-run the cells below.

## Initialization
rng_seed = 4137
rng_seed_init = 11 # seeds the random initial condition that gets spun up onto the attractor
nx = 3 # state dimension

## True (Lorenz 63) parameters and priors
sigma_true = 10.0
rho_true = 28.0
beta_true = 8.0 / 3.0
param_names = ["rho", "beta"]
prior_means = [20.0, 2.0]
prior_stds = [10.0, 1.0]
# Upper/lower bounds keep every algorithm (in particular Gauss-Newton, whose steps can be more
# aggressive) away from physically implausible parameter values that make the Lorenz 63
# integration numerically unstable.
prior_bounds = [(0, 100), (0, 20)] # (lo, hi) per parameter, matching param_names order

## Statistics windows
dt = 0.01 # timestep
T_spinup = 1000.0 # duration to spin up an initial condition onto the attractor
T = 40.0 # duration of a single forward run
T_start = 30.0 # discard the transient before collecting statistics
T_end = T
multiple = 36 # number of independent windows used to estimate the covariance Γy

## EKI / LM configuration
N_ens = 10 # number of EKI ensemble members
N_iter = 15 # number of EKI iterations
N_iter_lm = 15 # number of Levenberg-Marquardt iterations
eki_process = Inversion() # EKP process type
lm_lambda0 = 1.0 # initial LM damping parameter

data_names = ["l63_mean_cov"]

# %% Preliminaries
rng = MersenneTwister(rng_seed)
figure_save_directory, data_save_directory = make_output_dirs()
true_params_config = EnsembleMemberConfig(sigma_true, rho_true, beta_true)
params_true = [rho_true, beta_true]
n_param = length(param_names)
println(n_param)
println(params_true)

# %% Priors
priors = build_priors(param_names, prior_means, prior_stds, prior_bounds)

# %% Initial condition
x_initial, x0 = spin_up_attractor_ic(rng_seed_init, nx, true_params_config, dt, T_spinup)

# %% Generate (artificial) truth samples
# Note: The observables y are related to the parameters θ by: y = G(θ) + η
lorenz_config_settings, observation_config, y, ny, Γy, truth =
    generate_truth_and_covariance(true_params_config, x_initial, x0, dt, T, T_start, T_end, multiple, data_names)

# %% Calibrate with EKI
initial_params = construct_initial_ensemble(rng, priors, N_ens)

println("\nRunning EKI...")
ekiobj = EKP.EnsembleKalmanProcess(initial_params, truth, eki_process; rng = copy(rng))

err = Float64[]
for i in 1:N_iter
    params_i = get_ϕ_final(priors, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space
    g_ens = hcat(
        [
            lorenz_forward(
                EnsembleMemberConfig(sigma_true, params_i[1, j], params_i[2, j]),
                x0,
                lorenz_config_settings,
                observation_config,
            ) for j in 1:size(params_i, 2)
        ]...,
    )
    # `update_ensemble!` returns `nothing` after a normal step, or `true` if the
    # scheduler (e.g. `DataMisfitController`) decided to terminate early -- in the
    # latter case no new iteration is appended to the ensemble's history, so we must
    # stop looping too rather than keep requesting iterations that no longer exist.
    terminated = EKP.update_ensemble!(ekiobj, g_ens)
    if !isnothing(terminated)
        println("EKI terminated early before iteration " * string(i))
        break
    end
    push!(err, get_error(ekiobj)[end])
    println("Iteration: " * string(i) * ", Error: " * string(err[end]))
end

final_mean = get_ϕ_mean_final(priors, ekiobj)
println("EKI final estimate: " * string(final_mean) * ", true: " * string(params_true))

# %% Save EKI outputs
u_stored = get_u(ekiobj, return_array = false)
g_stored = get_g(ekiobj, return_array = false)

@save joinpath(data_save_directory, "l63_parameter_storage.jld2") u_stored
@save joinpath(data_save_directory, "l63_data_storage.jld2") g_stored
@save joinpath(data_save_directory, "l63_calibration_results.jld2") err final_mean params_true

# %% Compare: Levenberg-Marquardt (derivative-based), initialized at the prior mean.
# Jacobians are exact, via ForwardDiff through the full chaotic Lorenz solve --
# illustrating how a derivative-based approach behaves on this same problem.
R_inv_var = sqrt(inv(Symmetric(Γy)))
nu = n_param
G_lm = build_lm_forward_map(sigma_true, x0, lorenz_config_settings, observation_config)

println("\nRunning Levenberg-Marquardt...")
θ_lm_init = log.(prior_means) # start at the prior mean, parameterized as (log ρ, log β)
lm_history, lm_err = run_levenberg_marquardt(θ_lm_init, N_iter_lm, nu, R_inv_var, y, G_lm; λ0 = lm_lambda0)
println("LM final estimate: " * string(lm_history[end]) * ", true: " * string(params_true))

# %% Plot: error convergence
err_plot = plot_error_convergence(err, lm_err)
display(err_plot)
savefig(err_plot, joinpath(figure_save_directory, "l63_error_convergence.png"))

# %% Plot: recovered parameters vs truth
param_plot = plot_final_parameters(param_names, final_mean, params_true)
display(param_plot)
savefig(param_plot, joinpath(figure_save_directory, "l63_final_parameters.png"))

# %% Plot: recovered parameters vs truth (against the prior distributions)
p = plot(priors, size = (800, 400))
for (i, sp) in enumerate(p.subplots)
    vline!(sp, [params_true[i]], lc = "black", lw = 5, label = "ref")
    vline!(sp, [final_mean[i]], lc = "green", lw = 3, label = "EKI")
    vline!(sp, [lm_history[end][i]], lc = "magenta", lw = 3, label = "LM")
end
display(p)
savefig(p, joinpath(figure_save_directory, "l63_priors_with_estimates.png"))

# %% Plot: EKI ensemble convergence in (rho, beta) space -- initial vs. final only
n_eki_iterations = length(err)
convergence_plot = plot_eki_convergence(priors, ekiobj, n_eki_iterations, lm_history, params_true)
display(convergence_plot)
savefig(convergence_plot, joinpath(figure_save_directory, "l63_eki_convergence.png"))

# %% Plot: statistics window explanation
# Shows the full trajectory used for a calibration forward run, with the [T_start, T_end]
# window used to compute calibration statistics highlighted, for x/y/z in stacked panels.
stats_window_plot = plot_statistics_window(true_params_config, x0, dt, T, T_start, T_end)
display(stats_window_plot)
savefig(stats_window_plot, joinpath(figure_save_directory, "l63_statistics_window.png"))
