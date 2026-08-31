include("GModel_L63.jl") # Contains Lorenz 63 source code

# Import modules
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

rng_seed = 4137
rng = MersenneTwister(rng_seed)

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

###
###  Define the (true) parameters
###
# The classical Lorenz 63 parameter values (chaotic regime). Sigma is held fixed;
# rho and beta are the parameters to be learned.
sigma_true = 10.0
rho_true = 28.0
beta_true = 8.0 / 3.0
params_true = [rho_true, beta_true]
param_names = ["rho", "beta"]
n_param = length(param_names)

println(n_param)
println(params_true)

###
###  Define the parameter priors
###
prior_means = [20.0, 2.0]
prior_stds = [10.0, 1.0]
# Upper bounds keep every algorithm (in particular Gauss-Newton, whose steps can be more
# aggressive) away from physically implausible parameter values that make the Lorenz 63
# integration numerically unstable.
prior_rho = constrained_gaussian(param_names[1], prior_means[1], prior_stds[1], 0, 100)
prior_beta = constrained_gaussian(param_names[2], prior_means[2], prior_stds[2], 0, 20)
priors = combine_distributions([prior_rho, prior_beta])

###
###  Define the data from which we want to learn the parameters
###
data_names = ["l63_mean_cov"]

###
###  L63 model settings
###
nx = 3 # state dimension
dt = 0.01 # timestep
T_spinup = 1000.0 # duration to spin up an initial condition onto the attractor
T = 40.0 # duration of a single forward run
T_start = 30.0 # discard the transient before collecting statistics
T_end = T
lorenz_config_settings = LorenzConfig(dt, T)
observation_config = ObservationConfig(T_start, T_end)
true_params_config = EnsembleMemberConfig(sigma_true, rho_true, beta_true)

# Spin up an initial condition so that it lies on the attractor
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)
x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
x_spun_up = lorenz_solve(true_params_config, x_initial, LorenzConfig(dt, T_spinup))
x0 = x_spun_up[:, end]

###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(θ) + η
###
y = lorenz_forward(true_params_config, x0, lorenz_config_settings, observation_config)
ny = length(y)

# Compute internal variability covariance by sampling statistics over independent
# windows of a long trajectory at the true parameters
println("Using truth values to compute covariance")
multiple = 36
window = T_end - T_start
T_R = multiple * window + T_start
R_run = lorenz_solve(true_params_config, x_initial, LorenzConfig(dt, T_R))
R_sample_size = Int(ceil(multiple))
R_samples = zeros(ny, R_sample_size)
for ii in 1:R_sample_size
    local_obs_config = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
    R_samples[:, ii] = stats(R_run, LorenzConfig(dt, T_R), local_obs_config)
end
Γy = cov(R_samples, dims = 2)
println(Γy)

# Construct observation object
truth = Observation(Dict("samples" => y, "covariances" => Γy, "names" => data_names))

###
###  Calibrate: compare several Ensemble Kalman Process algorithms
###
N_ens = 10 # number of ensemble members
N_iter = 15 # number of iterations
initial_params = construct_initial_ensemble(rng, priors, N_ens)

println("\nRunning EKI...")
ekiobj = EKP.EnsembleKalmanProcess(initial_params, truth, Inversion(); rng = copy(rng))

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

u_stored = get_u(ekiobj, return_array = false)
g_stored = get_g(ekiobj, return_array = false)

@save data_save_directory * "l63_parameter_storage.jld2" u_stored
@save data_save_directory * "l63_data_storage.jld2" g_stored
@save data_save_directory * "l63_calibration_results.jld2" err final_mean params_true

###
###  Compare: Levenberg-Marquardt (derivative-based), initialized at the prior mean.
###  Jacobians are exact, via ForwardDiff through the full chaotic Lorenz solve --
###  illustrating how a derivative-based approach behaves on this same problem.
###
R_inv_var = sqrt(inv(Symmetric(Γy)))
nu = n_param

G_lm(θ) = lorenz_forward(
    EnsembleMemberConfig(promote(sigma_true, exp(θ[1]), exp(θ[2]))...),
    x0,
    lorenz_config_settings,
    observation_config,
)

function run_levenberg_marquardt(θ_lm, N_iter, nu, R_inv_var, y, G_lm)
    λ = 1.0
    ny_lm = length(y)
    lm_history = [exp.(θ_lm)]
    # Loss normalized as `1/ny * (y - G(θ))' * Γy⁻¹ * (y - G(θ))`, matching EKI's `get_error` (`compute_loss_at_mean`).
    lm_err = Float64[]
    for outer_iter in 1:N_iter
        r = y - G_lm(θ_lm)
        J = ForwardDiff.jacobian(G_lm, θ_lm)
        r̃ = R_inv_var * r
        J̃ = R_inv_var * J

        d = max.([norm(J̃[:, j]) for j in 1:nu], eps())
        A_aug = vcat(J̃, sqrt(λ) * Diagonal(d))
        b_aug = vcat(r̃, zeros(nu))
        Δθ = qr(A_aug, ColumnNorm()) \ b_aug

        θ_trial = θ_lm + Δθ
        r̃_trial = R_inv_var * (y - G_lm(θ_trial))

        gain = (norm(r̃)^2 - norm(r̃_trial)^2) / (norm(r̃)^2 - norm(J̃ * Δθ - r̃)^2)
        if gain > 0
            θ_lm = θ_trial
        end
        if !isfinite(gain) || gain < 0.25
            λ = min(λ * 4.0, 1e8)
        elseif gain > 0.75
            λ = max(λ / 3.0, 1e-10)
        end
        push!(lm_history, exp.(θ_lm))
        push!(lm_err, norm(gain > 0 ? r̃_trial : r̃)^2 / ny_lm)
        println("LM iteration: " * string(outer_iter) * ", params: " * string(exp.(θ_lm)))
    end
    return lm_history, lm_err
end

println("\nRunning Levenberg-Marquardt...")
θ_lm_init = log.(prior_means) # start at the prior mean, parameterized as (log ρ, log β)
lm_history, lm_err = run_levenberg_marquardt(θ_lm_init, N_iter, nu, R_inv_var, y, G_lm)
println("LM final estimate: " * string(lm_history[end]) * ", true: " * string(params_true))

###
###  Plot: error convergence
###
gr()
err_plot = plot(
    1:length(err),
    err,
    xlabel = "Iteration",
    ylabel = "Error",
    legend = :right,
    linewidth = 4,
    marker = :circle,
    color = :black,
    label = "EKI",
    size = (1280, 800),
    grid = false,
    guidefontsize = 24,
    tickfontsize = 18,
    legendfontsize = 21,
    left_margin = 10Plots.mm,
    bottom_margin = 10Plots.mm,
)
plot!(err_plot, 1:length(lm_err), lm_err, linewidth = 4, marker = :diamond, color = :magenta, label = "LM")
savefig(err_plot, joinpath(figure_save_directory, "l63_error_convergence.png"))

###
###  Plot: recovered parameters vs truth
###
param_plot = plot(layout = (1, n_param), size = (450 * n_param, 400), legend = false)
for (pp, pname) in enumerate(param_names)
    bar!(param_plot[pp], ["EKI"], [final_mean[pp]], title = pname)
    hline!(param_plot[pp], [params_true[pp]], linestyle = :dash, linecolor = :red)
end
savefig(param_plot, joinpath(figure_save_directory, "l63_final_parameters.png"))

###
###  Plot: EKI ensemble convergence in (rho, beta) space -- initial vs. final only
###
n_eki_iterations = length(err)
eki_color = :black
lm_color = :magenta

ϕ_init = get_ϕ(priors, ekiobj, 1)
ϕ_final = get_ϕ(priors, ekiobj, n_eki_iterations)

convergence_plot = scatter(
    ϕ_init[1, :],
    ϕ_init[2, :],
    color = eki_color,
    alpha = 0.4,
    markersize = 6,
    markerstrokewidth = 0,
    xlabel = "ρ",
    ylabel = "β",
    legend = false,
    size = (900, 700),
    dpi = 300,
    guidefontsize = 24,
    tickfontsize = 18,
    grid = false,
    left_margin = 10Plots.mm,
    bottom_margin = 10Plots.mm,
)
scatter!(
    convergence_plot,
    ϕ_final[1, :],
    ϕ_final[2, :],
    color = eki_color,
    alpha = 1.0,
    markersize = 6,
    markerstrokewidth = 0,
    label = false,
)
vline!(convergence_plot, [params_true[1]], linestyle = :dash, linecolor = :red)
hline!(convergence_plot, [params_true[2]], linestyle = :dash, linecolor = :red)

scatter!(
    convergence_plot,
    [lm_history[1][1]],
    [lm_history[1][2]],
    color = lm_color,
    alpha = 0.4,
    markershape = :cross,
    markersize = 9,
    markerstrokewidth = 4,
    markerstrokecolor = lm_color,
    label = false,
)
scatter!(
    convergence_plot,
    [lm_history[end][1]],
    [lm_history[end][2]],
    color = lm_color,
    alpha = 1.0,
    markershape = :cross,
    markersize = 9,
    markerstrokewidth = 4,
    markerstrokecolor = lm_color,
    label = false,
)
savefig(convergence_plot, joinpath(figure_save_directory, "l63_eki_convergence.png"))

###
###  Plot: animations of EKI ensemble convergence in (rho, beta) space, alongside the
###  x-z projection of the Lorenz 63 trajectories generated by the ensemble's
###  parameters at that iteration. The prior ensemble/trajectories are retained in
###  grey for reference on every frame, while the current iteration is drawn in
###  green. The LM comparison is omitted here.
###
prior_color = :grey
current_color = :green

ϕ_iters = [get_ϕ(priors, ekiobj, i) for i in 1:n_eki_iterations]
ϕ_prior = ϕ_iters[1]

traj_iters = [
    [lorenz_solve(EnsembleMemberConfig(sigma_true, ϕ_i[1, j], ϕ_i[2, j]), x0, lorenz_config_settings) for j in 1:N_ens] for ϕ_i in ϕ_iters
]
traj_prior = traj_iters[1]

ϕ_mean_iters = hcat([vec(mean(ϕ_i, dims = 2)) for ϕ_i in ϕ_iters]...)
mean_traj_iters = [
    lorenz_solve(EnsembleMemberConfig(sigma_true, ϕ_mean_iters[1, i], ϕ_mean_iters[2, i]), x0, lorenz_config_settings) for i in 1:n_eki_iterations
]
mean_traj_prior = mean_traj_iters[1]

# Fixed axis limits across the whole animation (computed over all iterations) so that
# frames don't rescale as the ensemble moves/collapses.
rho_lims = extrema(vcat([ϕ_i[1, :] for ϕ_i in ϕ_iters]...))
beta_lims = extrema(vcat([ϕ_i[2, :] for ϕ_i in ϕ_iters]...))
x_lims = extrema(vcat([xn[1, :] for traj in traj_iters for xn in traj]...))
z_lims = extrema(vcat([xn[3, :] for traj in traj_iters for xn in traj]...))

# Panel 1: EKI ensemble (green) vs. prior ensemble (grey) in (rho, beta) space, with
# the ensemble-mean path (black, triangle markers) traced up to iteration `i`.
function param_space_panel(i)
    panel = scatter(
        ϕ_prior[1, :],
        ϕ_prior[2, :],
        color = prior_color,
        alpha = 0.4,
        markersize = 6,
        markerstrokewidth = 0,
        xlabel = "ρ",
        ylabel = "β",
        xlims = rho_lims,
        ylims = beta_lims,
        legend = false,
        title = "Ensemble, iteration " * string(i - 1),
        guidefontsize = 14,
        tickfontsize = 10,
        top_margin = 5Plots.mm,
        left_margin = 8Plots.mm,
    )
    scatter!(
        panel,
        ϕ_iters[i][1, :],
        ϕ_iters[i][2, :],
        color = current_color,
        alpha = 1.0,
        markersize = 6,
        markerstrokewidth = 0,
    )
    vline!(panel, [params_true[1]], linestyle = :dash, linecolor = :red)
    hline!(panel, [params_true[2]], linestyle = :dash, linecolor = :red)
    plot!(
        panel,
        ϕ_mean_iters[1, 1:i],
        ϕ_mean_iters[2, 1:i],
        color = :black,
        linewidth = 2,
        marker = :utriangle,
        markersize = 7,
        markerstrokewidth = 0,
    )
    return panel
end

###
###  Animation 1: parameter-space panel on its own
###
param_anim = @animate for i in 1:n_eki_iterations
    param_space_panel(i)
end
gif(param_anim, joinpath(figure_save_directory, "l63_eki_paramspace_animation.gif"), fps = 2)

###
###  Animation 2: parameter-space panel + x-z projection of every ensemble member's trajectory
###
ensemble_traj_anim = @animate for i in 1:n_eki_iterations
    traj_panel = plot(
        xlabel = "x",
        ylabel = "z",
        xlims = x_lims,
        ylims = z_lims,
        legend = false,
        title = "Ensemble trajectories",
        guidefontsize = 14,
        tickfontsize = 10,
        top_margin = 5Plots.mm,
        left_margin = 8Plots.mm,
    )
    for xn in traj_prior
        plot!(traj_panel, xn[1, :], xn[3, :], color = prior_color, alpha = 0.3, linewidth = 1)
    end
    for xn in traj_iters[i]
        plot!(traj_panel, xn[1, :], xn[3, :], color = current_color, alpha = 0.6, linewidth = 1)
    end

    plot(
        param_space_panel(i),
        traj_panel,
        layout = (1, 2),
        size = (1600, 700),
        left_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm,
    )
end
gif(ensemble_traj_anim, joinpath(figure_save_directory, "l63_eki_convergence_animation.gif"), fps = 2)

###
###  Animation 3: parameter-space panel + x-z projection of only the ensemble-mean trajectory
###
mean_traj_anim = @animate for i in 1:n_eki_iterations
    traj_panel = plot(
        xlabel = "x",
        ylabel = "z",
        xlims = x_lims,
        ylims = z_lims,
        legend = false,
        title = "Mean-parameter trajectory",
        guidefontsize = 14,
        tickfontsize = 10,
        top_margin = 5Plots.mm,
        left_margin = 8Plots.mm,
    )
    plot!(traj_panel, mean_traj_prior[1, :], mean_traj_prior[3, :], color = prior_color, alpha = 0.6, linewidth = 1)
    plot!(traj_panel, mean_traj_iters[i][1, :], mean_traj_iters[i][3, :], color = :black, alpha = 1.0, linewidth = 1)

    plot(
        param_space_panel(i),
        traj_panel,
        layout = (1, 2),
        size = (1600, 700),
        left_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm,
    )
end
gif(mean_traj_anim, joinpath(figure_save_directory, "l63_eki_meantrajectory_animation.gif"), fps = 2)
