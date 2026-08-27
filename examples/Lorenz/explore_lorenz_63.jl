include("GModel_L63.jl") # Contains Lorenz 63 source code

using Plots
using Random
using LinearAlgebra
using Statistics

gr()

make_long_loss_plot = false # the 100τ_λ loss landscape (Plot 5) is expensive; toggle on to regenerate it
make_gif = true # animates Plot 1 + Plot 3 together; takes a little time, toggle off to skip

homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end

sigma_true = 10.0
rho_true = 28.0
beta_true = 8.0 / 3.0
classical_params = EnsembleMemberConfig(sigma_true, rho_true, beta_true)

dt = 0.01

# Largest Lyapunov exponent of the classical attractor (Viswanath, 1998)
lyapunov_exponent = 0.9056
lyapunov_time = 1.0 / lyapunov_exponent

rng = MersenneTwister(2399) # was 23
x_spinup = lorenz_solve(classical_params, [1.0, 1.0, 1.0], LorenzConfig(dt, 100.0))
x0_attractor = x_spinup[:, end]

n_runs = 10
param_perturbation_frac = 0.02

param_perturbations = [randn(rng, 2) for _ in 1:n_runs]
perturbed_params = [
    EnsembleMemberConfig(
        sigma_true,
        rho_true * (1 + param_perturbation_frac * s[1]),
        beta_true * (1 + param_perturbation_frac * s[2]),
    ) for s in param_perturbations
]

T_short = 5 * lyapunov_time
short_config = LorenzConfig(dt, T_short)
base_short_endpoint = lorenz_solve(classical_params, x0_attractor, short_config)[:, end]

target_short_sep =
    mean([norm(lorenz_solve(pp, x0_attractor, short_config)[:, end] - base_short_endpoint) for pp in perturbed_params])

# Solve for the IC perturbation size along `direction` matching the parameter perturbations' short-time separation.
function calibrate_perturbation_scale(direction, target_sep; cap = 1.0, max_iter = 40)
    sep(scale) = norm(lorenz_solve(classical_params, x0_attractor .+ scale .* direction, short_config)[:, end] .- base_short_endpoint)
    sep(cap) < target_sep && return nothing
    lo, hi = 0.0, cap
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        sep(mid) < target_sep ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

n_candidate_directions = 40
candidate_directions = [normalize(randn(rng, 3)) for _ in 1:n_candidate_directions]
calibrated_scales = [calibrate_perturbation_scale(d, target_short_sep) for d in candidate_directions]
valid = findall(!isnothing, calibrated_scales)
@assert length(valid) >= n_runs "not enough candidate directions reached the target short-time separation within a small perturbation; increase n_candidate_directions or cap"
chosen = valid[1:n_runs]
ic_directions = candidate_directions[chosen]
ic_scales = Float64.(calibrated_scales[chosen])
println("Calibrated initial-condition perturbation sizes: ", ic_scales)
x0_ic = [x0_attractor .+ s .* d for (s, d) in zip(ic_scales, ic_directions)]

runs = vcat([(x0_ic[i], classical_params) for i in 1:n_runs], [(x0_attractor, pp) for pp in perturbed_params])
colors = vcat(fill(:steelblue, n_runs), fill(:orangered, n_runs))
labels = vcat(["IC perturbation $i" for i in 1:n_runs], ["Parameter perturbation $i" for i in 1:n_runs])
group_labels = [i == 1 ? "IC perturbation" : (i == n_runs + 1 ? "Parameter perturbation" : false) for i in 1:(2 * n_runs)]

# Plot 1: short-time trajectories -- IC and parameter perturbations look indistinguishable.
short_trajectories = [lorenz_solve(p, x0, short_config) for (x0, p) in runs]

p_short = plot(
    xlabel = "x",
    ylabel = "y",
    zlabel = "z",
    legend = :best,
    size = (1200, 900),
    dpi = 300,
    guidefontsize = 24,
    tickfontsize = 18,
    legendfontsize = 21,
    grid = false,
)
for (i, xn) in enumerate(short_trajectories)
    plot!(p_short, xn[1, :], xn[2, :], xn[3, :], color = colors[i], label = group_labels[i], linewidth = 3)
    scatter!(p_short, [xn[1, end]], [xn[2, end]], [xn[3, end]], color = colors[i], markershape = :utriangle, markersize = 8, label = false)
end
reference_short = lorenz_solve(classical_params, x0_attractor, short_config)
plot!(p_short, reference_short[1, :], reference_short[2, :], reference_short[3, :], color = :black, label = "Reference", linewidth = 3)
scatter!(p_short, [reference_short[1, 1]], [reference_short[2, 1]], [reference_short[3, 1]], color = :black, markershape = :circle, markersize = 8, label = false)
scatter!(p_short, [reference_short[1, end]], [reference_short[2, end]], [reference_short[3, end]], color = :black, markershape = :utriangle, markersize = 8, label = false)
p_short_path = joinpath(figure_save_directory, "l63_short_time_trajectories.png")
savefig(p_short, p_short_path)
@info "Saved short-time trajectories figure to $(p_short_path)"

# Plot 2: long-time butterflies and energy -- unrelated ICs leave the statistic unchanged, perturbed params shift it.
T_energy_short = 1.0 * lyapunov_time
T_medium = 10.0 * lyapunov_time
T_long = 100.0 * lyapunov_time

x0_far = [lorenz_solve(classical_params, randn(rng, 3), LorenzConfig(dt, 100.0))[:, end] for _ in 1:n_runs]

runs_long =
    vcat([(x0_far[i], classical_params) for i in 1:n_runs], [(x0_far[i], perturbed_params[i]) for i in 1:n_runs])
labels_long = vcat(["Unrelated IC $i" for i in 1:n_runs], ["Parameter perturbation $i" for i in 1:n_runs])

long_trajectories = [lorenz_solve(p, x0, LorenzConfig(dt, T_long)) for (x0, p) in runs_long]

butterfly_plots = [
    plot(
        xn[1, :],
        xn[2, :],
        xn[3, :],
        color = colors[i],
        label = false,
        xlabel = "",
        ylabel = "",
        zlabel = "",
        xticks = false,
        yticks = false,
        zticks = false,
        grid = false,
        linewidth = 0.6,
    ) for (i, xn) in enumerate(long_trajectories)
]
n_butterfly_cols = 3
butterfly_indices = vcat(1:n_butterfly_cols, (n_runs + 1):(n_runs + n_butterfly_cols))
legend_strip = plot(
    [NaN], [NaN],
    color = :steelblue,
    label = "different IC",
    linewidth = 3,
    legend = :top,
    legendfontsize = 21,
    legend_column = 2,
    framestyle = :none,
    grid = false,
)
plot!(legend_strip, [NaN], [NaN], color = :orangered, label = "different parameters", linewidth = 3)
grid_layout = @layout [a{0.08h}; grid(2, 3)]
grid_plot = plot(
    legend_strip,
    butterfly_plots[butterfly_indices]...,
    layout = grid_layout,
    size = (400 * n_butterfly_cols, 860),
)
grid_plot_path = joinpath(figure_save_directory, "l63_long_time_butterflies.png")
savefig(grid_plot, grid_plot_path)
@info "Saved long-time butterflies figure to $(grid_plot_path)"

function energy_integral(xn, dt)
    T = dt * (size(xn, 2) - 1)
    e = dropdims(sum(xn .^ 2, dims = 1), dims = 1)
    return (dt * (sum(e) - 0.5 * (e[1] + e[end]))) / T
end
E_values = [energy_integral(xn, dt) for xn in long_trajectories]
short_window_of_long_runs = [lorenz_solve(p, x0, LorenzConfig(dt, T_energy_short)) for (x0, p) in runs_long]
E_short_values = [energy_integral(xn, dt) for xn in short_window_of_long_runs]
medium_window_of_long_runs = [lorenz_solve(p, x0, LorenzConfig(dt, T_medium)) for (x0, p) in runs_long]
E_medium_values = [energy_integral(xn, dt) for xn in medium_window_of_long_runs]

E_ic_by_duration = [E_short_values[1:n_runs], E_medium_values[1:n_runs], E_values[1:n_runs]]
E_param_by_duration =
    [E_short_values[(n_runs + 1):(2 * n_runs)], E_medium_values[(n_runs + 1):(2 * n_runs)], E_values[(n_runs + 1):(2 * n_runs)]]
dodge = 0.15

energy_plot = scatter(
    fill(1 - dodge, n_runs),
    E_ic_by_duration[1],
    color = :steelblue,
    markersize = 8,
    markerstrokewidth = 0,
    label = "Initial conditions",
    xticks = (1:3, ["1τ", "10τ", "100τ"]),
    xlims = (0.5, 3.5),
    xlabel = "Trajectory length",
    ylabel = "Average size of trajectory",
    grid = false,
    legend = false,
)
scatter!(energy_plot, fill(1 + dodge, n_runs), E_param_by_duration[1], color = :orangered, markersize = 8, markerstrokewidth = 0, label = "Parameters")
for k in 2:3
    scatter!(energy_plot, fill(k - dodge, n_runs), E_ic_by_duration[k], color = :steelblue, markersize = 8, markerstrokewidth = 0, label = false)
    scatter!(energy_plot, fill(k + dodge, n_runs), E_param_by_duration[k], color = :orangered, markersize = 8, markerstrokewidth = 0, label = false)
end
energy_plot_path = joinpath(figure_save_directory, "l63_long_time_energy.png")
savefig(energy_plot, energy_plot_path)
@info "Saved long-time energy figure to $(energy_plot_path)"

println("Energy values (different initial conditions, classical params): ", E_values[1:n_runs])
println("Energy values (parameter perturbations): ", E_values[(n_runs + 1):(2 * n_runs)])

# Plot 3: separation from a reference trajectory over time -- makes plot 1's "indistinguishable" claim quantitative.
# Since the true IC is unknown, the parameter-perturbation group also starts from the perturbed x0_ic values.
T_growth = 8.0 * lyapunov_time
growth_config = LorenzConfig(dt, T_growth)
reference_trajectory = lorenz_solve(classical_params, x0_attractor, growth_config)
runs_sep = vcat([(x0_ic[i], classical_params) for i in 1:n_runs], [(x0_ic[i], perturbed_params[i]) for i in 1:n_runs])
growth_trajectories = [lorenz_solve(p, x0, growth_config) for (x0, p) in runs_sep]
t_axis = (0:(size(reference_trajectory, 2) - 1)) .* dt ./ lyapunov_time

separation_plot = plot(
    xlabel = "Time (Lyapunov times)",
    ylabel = "Separation from reference trajectory",
    yscale = :log10,
    legend = false,
    size = (1280, 800),
    grid = false,
    guidefontsize = 24,
    tickfontsize = 18,
    left_margin = 20Plots.mm,
    bottom_margin = 20Plots.mm,
)
for (i, xn) in enumerate(growth_trajectories)
    sep_t = clamp.([norm(xn[:, k] .- reference_trajectory[:, k]) for k in 1:size(xn, 2)], 1e-12, Inf)
    plot!(separation_plot, t_axis, sep_t, color = colors[i], label = false, linewidth = 3)
end
vline!(separation_plot, [T_short / lyapunov_time], linestyle = :dash, linecolor = :black, label = false)
separation_plot_path = joinpath(figure_save_directory, "l63_separation_growth.png")
savefig(separation_plot, separation_plot_path)
@info "Saved separation growth figure to $(separation_plot_path)"

# Animation: Plot 1's trajectories (left) and Plot 3's separation growth (right) unfolding together,
# both run out over the full T_growth window so the two panels stay in sync throughout.
# This takes a little time to render; toggle off with make_gif = false to skip it.
if make_gif
    long_trajectories_p1 = [lorenz_solve(p, x0, growth_config) for (x0, p) in runs]

    sep_curves = [
        clamp.([norm(xn[:, k] .- reference_trajectory[:, k]) for k in 1:size(xn, 2)], 1e-12, Inf) for
        xn in growth_trajectories
    ]
    sep_ceil = 10^ceil(log10(maximum(reduce(vcat, sep_curves))))
    sep_ylims = (1e-3, sep_ceil)

    traj_xlims = extrema(vcat([xn[1, :] for xn in long_trajectories_p1]..., reference_trajectory[1, :]))
    traj_ylims = extrema(vcat([xn[2, :] for xn in long_trajectories_p1]..., reference_trajectory[2, :]))
    traj_zlims = extrema(vcat([xn[3, :] for xn in long_trajectories_p1]..., reference_trajectory[3, :]))

    n_frames = 200
    anim_duration = 10.0 # seconds
    frame_indices = round.(Int, range(1, size(reference_trajectory, 2), length = n_frames))

    anim = @animate for idx_g in frame_indices
        traj_panel = plot(
            xlabel = "x",
            ylabel = "y",
            zlabel = "z",
            xlims = traj_xlims,
            ylims = traj_ylims,
            zlims = traj_zlims,
            title = "Trajectories",
            titlefontsize = 28,
            legend = :topleft,
            legendfontsize = 16,
            guidefontsize = 24,
            tickfontsize = 16,
            grid = false,
        )
        for (i, xn) in enumerate(long_trajectories_p1)
            plot!(traj_panel, xn[1, 1:idx_g], xn[2, 1:idx_g], xn[3, 1:idx_g], color = colors[i], label = group_labels[i], linewidth = 2)
        end
        plot!(
            traj_panel,
            reference_trajectory[1, 1:idx_g],
            reference_trajectory[2, 1:idx_g],
            reference_trajectory[3, 1:idx_g],
            color = :black,
            label = "Reference",
            linewidth = 4,
        )

        sep_panel = plot(
            ylabel = "Separation from reference trajectory",
            yscale = :log10,
            xlims = (0, t_axis[end]),
            ylims = sep_ylims,
            title = "Separation growth",
            titlefontsize = 28,
            legend = false,
            guidefontsize = 24,
            tickfontsize = 16,
            grid = false,
        )
        for (i, sep_t) in enumerate(sep_curves)
            plot!(sep_panel, t_axis[1:idx_g], sep_t[1:idx_g], color = colors[i], linewidth = 2)
        end

        plot(traj_panel, sep_panel, layout = (1, 2), size = (1600, 700))
    end
    gif_path = joinpath(figure_save_directory, "l63_trajectories_and_separation.gif")
    gif(anim, gif_path, fps = round(Int, n_frames / anim_duration))
    @info "Saved trajectories + separation animation to $(gif_path)"
end

# Plot 4: 2D loss surface |E(θ) - E_true| over (ρ, β) -- one continuous simulation, chaining a
# short spin-up (not a cold restart) into each parameter's 10τ_λ statistics window.
T_spinup_short = 2.0 * lyapunov_time
n_grid = 100
rho_range = range(0.95 * rho_true, 1.05 * rho_true, length = n_grid)
beta_range = range(0.8 * beta_true, 1.2 * beta_true, length = n_grid)

reference_medium = lorenz_solve(classical_params, x0_attractor, LorenzConfig(dt, T_medium))
E_ref = energy_integral(reference_medium, dt)

function compute_loss_grid(rho_range, beta_range, x0_start, T_spinup_short, T_stats, E_ref)
    state = x0_start
    loss_grid = zeros(length(rho_range), length(beta_range))
    for (i, rho_g) in enumerate(rho_range)
        for (j, beta_g) in enumerate(beta_range)
            params_g = EnsembleMemberConfig(sigma_true, rho_g, beta_g)
            state = lorenz_solve(params_g, state, LorenzConfig(dt, T_spinup_short))[:, end]
            stats_traj = lorenz_solve(params_g, state, LorenzConfig(dt, T_stats))
            state = stats_traj[:, end]
            loss_grid[i, j] = abs(energy_integral(stats_traj, dt) - E_ref)
        end
    end
    return loss_grid
end
loss_grid = compute_loss_grid(rho_range, beta_range, x0_ic[1], T_spinup_short, T_medium, E_ref)

loss_plot = surface(
    rho_range,
    beta_range,
    loss_grid',
    xlabel = "ρ",
    ylabel = "β",
    zlabel = "loss",
    size = (1200, 900),
    dpi = 300,
    legend = false,
    guidefontsize = 24,
    tickfontsize = 18,
    grid = false,
)
scatter!(loss_plot, [rho_true], [beta_true], [0.0], color = :white, markershape = :star5, markersize = 16, markerstrokecolor = :black, label = false)
loss_plot_path = joinpath(figure_save_directory, "l63_loss_landscape.png")
savefig(loss_plot, loss_plot_path)
@info "Saved loss landscape figure to $(loss_plot_path)"

# Plot 5: same loss landscape, but from 100τ_λ trajectories -- the longer statistics window averages
# out more of the chaotic noise, so the surface should look markedly smoother than the 10τ_λ version.
if make_long_loss_plot
    reference_long_run = lorenz_solve(classical_params, x0_attractor, LorenzConfig(dt, T_long))
    E_ref_long = energy_integral(reference_long_run, dt)
    loss_grid_long = compute_loss_grid(rho_range, beta_range, x0_ic[1], T_spinup_short, T_long, E_ref_long)

    loss_plot_long = surface(
        rho_range,
        beta_range,
        loss_grid_long',
        xlabel = "ρ",
        ylabel = "β",
        zlabel = "loss",
        size = (1200, 900),
        dpi = 300,
        legend = false,
        guidefontsize = 24,
        tickfontsize = 18,
        grid = false,
    )
    scatter!(
        loss_plot_long,
        [rho_true],
        [beta_true],
        [0.0],
        color = :white,
        markershape = :star5,
        markersize = 16,
        markerstrokecolor = :black,
        label = false,
    )
    loss_plot_long_path = joinpath(figure_save_directory, "l63_loss_landscape_long.png")
    savefig(loss_plot_long, loss_plot_long_path)
    @info "Saved long loss landscape figure to $(loss_plot_long_path)"
end
