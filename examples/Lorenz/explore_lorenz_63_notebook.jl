# %% Imports
using Plots
using Random
using LinearAlgebra
using Statistics

include("GModel_L63.jl") # Contains Lorenz 63 source code
include("explore_lorenz_63_notebook_utils.jl") # Preliminaries / data construction / plotting helpers

gr()

# %% Tunable parameters
# Play with any of these and re-run the cells below.

make_long_loss_plot = false # the 100τ_λ loss landscape (Plot 5) is expensive; toggle on to regenerate it
make_gif = true # animates Plot 1 + Plot 3 together; takes a little time, toggle off to skip

## Initialization
rng_seed = 2399
sigma_true = 10.0
rho_true = 28.0
beta_true = 8.0 / 3.0
dt = 0.01
lyapunov_exponent = 0.9056 # largest Lyapunov exponent of the classical attractor (Viswanath, 1998)

## Perturbations
n_runs = 10
param_perturbation_frac = 0.02
n_candidate_directions = 40 # candidate IC-perturbation directions searched to match the parameter perturbations' short-time separation

## Statistics windows (multiples of the Lyapunov time)
T_short_mult = 5.0 # Plot 1 / short-time trajectories
T_energy_short_mult = 1.0 # Plot 2 / shortest energy window
T_medium_mult = 10.0 # Plot 2 / medium energy window, also used for the loss landscape
T_long_mult = 100.0 # Plot 2 / longest energy window
T_growth_mult = 8.0 # Plot 3 + animation / separation growth window
T_spinup_short_mult = 2.0 # Plot 4/5 / spin-up between loss-grid points

## Plot / animation configuration
n_butterfly_cols = 3 # Plot 2 butterfly grid columns per group
n_grid = 100 # loss-landscape (ρ, β) grid resolution
n_frames = 200 # animation frame count
anim_duration = 10.0 # animation duration in seconds

# %% Preliminaries
figure_save_directory = make_output_dir()
classical_params = EnsembleMemberConfig(sigma_true, rho_true, beta_true)
lyapunov_time = 1.0 / lyapunov_exponent
rng = MersenneTwister(rng_seed)
x0_attractor = spin_up_x0_attractor(classical_params, dt)
colors, group_labels = run_group_style(n_runs)

# %% Matched-magnitude IC vs. parameter perturbations
perturbed_params = build_perturbed_params(rng, n_runs, sigma_true, rho_true, beta_true, param_perturbation_frac)
T_short = T_short_mult * lyapunov_time
short_config = LorenzConfig(dt, T_short)
x0_ic = calibrate_ic_perturbations(
    rng,
    classical_params,
    x0_attractor,
    perturbed_params,
    short_config,
    n_runs;
    n_candidate_directions = n_candidate_directions,
)

# %% Plot 1: short-time trajectories -- IC and parameter perturbations look indistinguishable.
runs = build_runs(x0_ic, x0_attractor, classical_params, perturbed_params, n_runs)
short_trajectories = solve_group(runs, short_config)
reference_short = lorenz_solve(classical_params, x0_attractor, short_config)

p_short = plot_short_time_trajectories(short_trajectories, reference_short, colors, group_labels)
display(p_short)
p_short_path = joinpath(figure_save_directory, "l63_short_time_trajectories.png")
savefig(p_short, p_short_path)
@info "Saved short-time trajectories figure to $(p_short_path)"

# %% Plot 2: long-time butterflies and energy -- unrelated ICs leave the statistic unchanged, perturbed params shift it.
T_energy_short = T_energy_short_mult * lyapunov_time
T_medium = T_medium_mult * lyapunov_time
T_long = T_long_mult * lyapunov_time

runs_long = build_unrelated_ic_runs(rng, classical_params, perturbed_params, n_runs, dt)
long_trajectories = solve_group(runs_long, LorenzConfig(dt, T_long))

grid_plot = plot_long_time_butterflies(long_trajectories, colors, n_runs, n_butterfly_cols)
display(grid_plot)
grid_plot_path = joinpath(figure_save_directory, "l63_long_time_butterflies.png")
savefig(grid_plot, grid_plot_path)
@info "Saved long-time butterflies figure to $(grid_plot_path)"

E_values = [energy_integral(xn, dt) for xn in long_trajectories]
short_window_of_long_runs = solve_group(runs_long, LorenzConfig(dt, T_energy_short))
E_short_values = [energy_integral(xn, dt) for xn in short_window_of_long_runs]
medium_window_of_long_runs = solve_group(runs_long, LorenzConfig(dt, T_medium))
E_medium_values = [energy_integral(xn, dt) for xn in medium_window_of_long_runs]

E_ic_by_duration = [E_short_values[1:n_runs], E_medium_values[1:n_runs], E_values[1:n_runs]]
E_param_by_duration = [
    E_short_values[(n_runs + 1):(2 * n_runs)],
    E_medium_values[(n_runs + 1):(2 * n_runs)],
    E_values[(n_runs + 1):(2 * n_runs)],
]
duration_labels = ["$(Int(T_energy_short_mult))τ", "$(Int(T_medium_mult))τ", "$(Int(T_long_mult))τ"]

energy_plot = plot_long_time_energy(E_ic_by_duration, E_param_by_duration, n_runs, duration_labels)
display(energy_plot)
energy_plot_path = joinpath(figure_save_directory, "l63_long_time_energy.png")
savefig(energy_plot, energy_plot_path)
@info "Saved long-time energy figure to $(energy_plot_path)"

println("Energy values (different initial conditions, classical params): ", E_values[1:n_runs])
println("Energy values (parameter perturbations): ", E_values[(n_runs + 1):(2 * n_runs)])

# %% Plot 3: separation from a reference trajectory over time -- makes plot 1's "indistinguishable" claim quantitative.
# Since the true IC is unknown, the parameter-perturbation group also starts from the perturbed x0_ic values.
T_growth = T_growth_mult * lyapunov_time
growth_config = LorenzConfig(dt, T_growth)
reference_trajectory = lorenz_solve(classical_params, x0_attractor, growth_config)
runs_sep = build_runs_sep(x0_ic, classical_params, perturbed_params, n_runs)
growth_trajectories = solve_group(runs_sep, growth_config)
t_axis = (0:(size(reference_trajectory, 2) - 1)) .* dt ./ lyapunov_time

separation_plot = plot_separation_growth(t_axis, growth_trajectories, reference_trajectory, colors, T_short, lyapunov_time)
display(separation_plot)
separation_plot_path = joinpath(figure_save_directory, "l63_separation_growth.png")
savefig(separation_plot, separation_plot_path)
@info "Saved separation growth figure to $(separation_plot_path)"

# %% Animation: Plot 1's trajectories (left) and Plot 3's separation growth (right) unfolding together,
# both run out over the full T_growth window so the two panels stay in sync throughout.
# This takes a little time to render; toggle off with make_gif = false to skip it.
if make_gif
    long_trajectories_p1 = solve_group(runs, growth_config)
    anim = build_trajectory_separation_animation(long_trajectories_p1, reference_trajectory, growth_trajectories, t_axis, colors, group_labels, n_frames)
    gif_path = joinpath(figure_save_directory, "l63_trajectories_and_separation.gif")
    gif(anim, gif_path, fps = round(Int, n_frames / anim_duration))
end

# %% Plot 4: 2D loss surface |E(θ) - E_true| over (ρ, β) -- one continuous simulation, chaining a
# short spin-up (not a cold restart) into each parameter's statistics window.
T_spinup_short = T_spinup_short_mult * lyapunov_time
rho_range = range(0.95 * rho_true, 1.05 * rho_true, length = n_grid)
beta_range = range(0.8 * beta_true, 1.2 * beta_true, length = n_grid)

reference_medium = lorenz_solve(classical_params, x0_attractor, LorenzConfig(dt, T_medium))
E_ref = energy_integral(reference_medium, dt)
loss_grid = compute_loss_grid(rho_range, beta_range, x0_ic[1], sigma_true, T_spinup_short, T_medium, E_ref, dt)

loss_plot = plot_loss_landscape(rho_range, beta_range, loss_grid, rho_true, beta_true)
display(loss_plot)
loss_plot_path = joinpath(figure_save_directory, "l63_loss_landscape.png")
savefig(loss_plot, loss_plot_path)
@info "Saved loss landscape figure to $(loss_plot_path)"

# %% Plot 5: same loss landscape, but from 100τ_λ trajectories -- the longer statistics window averages
# out more of the chaotic noise, so the surface should look markedly smoother than the 10τ_λ version.
if make_long_loss_plot
    reference_long_run = lorenz_solve(classical_params, x0_attractor, LorenzConfig(dt, T_long))
    E_ref_long = energy_integral(reference_long_run, dt)
    loss_grid_long = compute_loss_grid(rho_range, beta_range, x0_ic[1], sigma_true, T_spinup_short, T_long, E_ref_long, dt)

    loss_plot_long = plot_loss_landscape(rho_range, beta_range, loss_grid_long, rho_true, beta_true)
    display(loss_plot_long)
    loss_plot_long_path = joinpath(figure_save_directory, "l63_loss_landscape_long.png")
    savefig(loss_plot_long, loss_plot_long_path)
    @info "Saved long loss landscape figure to $(loss_plot_long_path)"
end
