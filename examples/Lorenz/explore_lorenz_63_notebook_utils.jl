# Supporting functions for explore_lorenz_63_notebook.jl: filesystem/preliminaries, data
# construction, and plotting. Assumes GModel_L63.jl and the required packages are already
# loaded by the caller.

###
### Preliminaries / filesystem
###

function make_output_dir(root_dir::AbstractString = pwd(); figure_subdir::AbstractString = "output")
    println(root_dir)
    figure_save_directory = joinpath(root_dir, figure_subdir)
    isdir(figure_save_directory) || mkdir(figure_save_directory)
    return figure_save_directory
end

function spin_up_x0_attractor(classical_params, dt; T_spinup = 100.0, x0_init = [1.0, 1.0, 1.0])
    x_spinup = lorenz_solve(classical_params, x0_init, LorenzConfig(dt, T_spinup))
    return x_spinup[:, end]
end

###
### Data construction
###

solve_group(runs, config) = [lorenz_solve(p, x0, config) for (x0, p) in runs]

function build_perturbed_params(rng, n_runs, sigma_true, rho_true, beta_true, param_perturbation_frac)
    param_perturbations = [randn(rng, 2) for _ in 1:n_runs]
    return [
        EnsembleMemberConfig(
            sigma_true,
            rho_true * (1 + param_perturbation_frac * s[1]),
            beta_true * (1 + param_perturbation_frac * s[2]),
        ) for s in param_perturbations
    ]
end

# Solve for the IC perturbation size along `direction` matching the parameter perturbations' short-time separation.
function calibrate_perturbation_scale(
    direction,
    target_sep,
    classical_params,
    x0_attractor,
    short_config,
    base_short_endpoint;
    cap = 1.0,
    max_iter = 40,
)
    sep(scale) = norm(
        lorenz_solve(classical_params, x0_attractor .+ scale .* direction, short_config)[:, end] .- base_short_endpoint,
    )
    sep(cap) < target_sep && return nothing
    lo, hi = 0.0, cap
    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        sep(mid) < target_sep ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

# Finds `n_runs` initial-condition perturbations whose short-time (`short_config`) separation from
# the reference trajectory matches the parameter perturbations' short-time separation, so that Plot 1
# compares IC and parameter perturbations of matched initial magnitude.
function calibrate_ic_perturbations(
    rng,
    classical_params,
    x0_attractor,
    perturbed_params,
    short_config,
    n_runs;
    n_candidate_directions = 40,
)
    base_short_endpoint = lorenz_solve(classical_params, x0_attractor, short_config)[:, end]
    target_short_sep = mean([
        norm(lorenz_solve(pp, x0_attractor, short_config)[:, end] - base_short_endpoint) for pp in perturbed_params
    ])

    candidate_directions = [normalize(randn(rng, 3)) for _ in 1:n_candidate_directions]
    calibrated_scales = [
        calibrate_perturbation_scale(
            d,
            target_short_sep,
            classical_params,
            x0_attractor,
            short_config,
            base_short_endpoint,
        ) for d in candidate_directions
    ]
    valid = findall(!isnothing, calibrated_scales)
    @assert length(valid) >= n_runs "not enough candidate directions reached the target short-time separation within a small perturbation; increase n_candidate_directions or cap"
    chosen = valid[1:n_runs]
    ic_directions = candidate_directions[chosen]
    ic_scales = Float64.(calibrated_scales[chosen])
    println("Calibrated initial-condition perturbation sizes: ", ic_scales)
    return [x0_attractor .+ s .* d for (s, d) in zip(ic_scales, ic_directions)]
end

# Shared color/legend-label convention used across all plots: the first `n_runs` entries are the
# IC-perturbation group (steelblue), the second `n_runs` are the parameter-perturbation group (orangered).
function run_group_style(n_runs)
    colors = vcat(fill(:steelblue, n_runs), fill(:orangered, n_runs))
    group_labels =
        [i == 1 ? "IC perturbation" : (i == n_runs + 1 ? "Parameter perturbation" : false) for i in 1:(2 * n_runs)]
    return colors, group_labels
end

# Plot 1's runs: IC perturbations (varying x0, true params) vs. parameter perturbations (fixed x0_attractor).
function build_runs(x0_ic, x0_attractor, classical_params, perturbed_params, n_runs)
    return vcat([(x0_ic[i], classical_params) for i in 1:n_runs], [(x0_attractor, pp) for pp in perturbed_params])
end

# Plot 3 / animation's runs: since the true IC is unknown, the parameter-perturbation group also
# starts from the perturbed x0_ic values (rather than the unperturbed x0_attractor).
function build_runs_sep(x0_ic, classical_params, perturbed_params, n_runs)
    return vcat([(x0_ic[i], classical_params) for i in 1:n_runs], [(x0_ic[i], perturbed_params[i]) for i in 1:n_runs])
end

# Plot 2's runs: unrelated initial conditions (each spun up from an independent random start) vs.
# parameter perturbations, isolating the effect of the parameter change on long-time statistics.
function build_unrelated_ic_runs(rng, classical_params, perturbed_params, n_runs, dt; T_spinup = 100.0)
    x0_far = [lorenz_solve(classical_params, randn(rng, 3), LorenzConfig(dt, T_spinup))[:, end] for _ in 1:n_runs]
    return vcat([(x0_far[i], classical_params) for i in 1:n_runs], [(x0_far[i], perturbed_params[i]) for i in 1:n_runs])
end

function energy_integral(xn, dt)
    T = dt * (size(xn, 2) - 1)
    e = dropdims(sum(xn .^ 2, dims = 1), dims = 1)
    return (dt * (sum(e) - 0.5 * (e[1] + e[end]))) / T
end

# One continuous simulation per grid point, chaining a short spin-up (not a cold restart) into
# each parameter's statistics window.
function compute_loss_grid(rho_range, beta_range, x0_start, sigma_true, T_spinup_short, T_stats, E_ref, dt)
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

###
### Plotting
###

# Plot 1: short-time trajectories -- IC and parameter perturbations look indistinguishable.
function plot_short_time_trajectories(short_trajectories, reference_short, colors, group_labels)
    p_short = plot(
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        legend = :best,
        size = (800, 600),
        guidefontsize = 24,
        tickfontsize = 18,
        legendfontsize = 21,
        grid = false,
    )
    for (i, xn) in enumerate(short_trajectories)
        plot!(p_short, xn[1, :], xn[2, :], xn[3, :], color = colors[i], label = group_labels[i], linewidth = 3)
        scatter!(
            p_short,
            [xn[1, end]],
            [xn[2, end]],
            [xn[3, end]],
            color = colors[i],
            markershape = :utriangle,
            markersize = 8,
            label = false,
        )
    end
    plot!(
        p_short,
        reference_short[1, :],
        reference_short[2, :],
        reference_short[3, :],
        color = :black,
        label = "Reference",
        linewidth = 3,
    )
    scatter!(
        p_short,
        [reference_short[1, 1]],
        [reference_short[2, 1]],
        [reference_short[3, 1]],
        color = :black,
        markershape = :circle,
        markersize = 8,
        label = false,
    )
    scatter!(
        p_short,
        [reference_short[1, end]],
        [reference_short[2, end]],
        [reference_short[3, end]],
        color = :black,
        markershape = :utriangle,
        markersize = 8,
        label = false,
    )
    return p_short
end

# Plot 2 (part 1): long-time butterflies -- unrelated ICs leave the statistic unchanged, perturbed params shift it.
function plot_long_time_butterflies(long_trajectories, colors, n_runs, n_butterfly_cols)
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
    butterfly_indices = vcat(1:n_butterfly_cols, (n_runs + 1):(n_runs + n_butterfly_cols))
    legend_strip = plot(
        [NaN],
        [NaN],
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
    grid_layout = @layout [a{0.08h}; grid(2, n_butterfly_cols)]
    return plot(legend_strip, butterfly_plots[butterfly_indices]..., layout = grid_layout, size = (800, 573))
end

# Plot 2 (part 2): average trajectory size ("energy") vs. statistics-window duration.
function plot_long_time_energy(E_ic_by_duration, E_param_by_duration, n_runs, duration_labels)
    dodge = 0.15
    energy_plot = scatter(
        fill(1 - dodge, n_runs),
        E_ic_by_duration[1],
        color = :steelblue,
        markersize = 8,
        markerstrokewidth = 0,
        label = "Initial conditions",
        xticks = (1:3, duration_labels),
        xlims = (0.5, 3.5),
        xlabel = "Trajectory length",
        ylabel = "Average size of trajectory",
        size = (800, 500),
        grid = false,
        legend = false,
    )
    scatter!(
        energy_plot,
        fill(1 + dodge, n_runs),
        E_param_by_duration[1],
        color = :orangered,
        markersize = 8,
        markerstrokewidth = 0,
        label = "Parameters",
    )
    for k in 2:3
        scatter!(
            energy_plot,
            fill(k - dodge, n_runs),
            E_ic_by_duration[k],
            color = :steelblue,
            markersize = 8,
            markerstrokewidth = 0,
            label = false,
        )
        scatter!(
            energy_plot,
            fill(k + dodge, n_runs),
            E_param_by_duration[k],
            color = :orangered,
            markersize = 8,
            markerstrokewidth = 0,
            label = false,
        )
    end
    return energy_plot
end

# Plot 3: separation from a reference trajectory over time -- makes plot 1's "indistinguishable"
# claim quantitative. The dashed line marks where Plot 1's short-time window ends.
function plot_separation_growth(t_axis, growth_trajectories, reference_trajectory, colors, T_short, lyapunov_time)
    separation_plot = plot(
        xlabel = "Time (Lyapunov times)",
        ylabel = "Separation from reference trajectory",
        yscale = :log10,
        legend = false,
        size = (800, 500),
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
    return separation_plot
end

# Animation: Plot 1's trajectories (left) and Plot 3's separation growth (right) unfolding together,
# both run out over the full growth window so the two panels stay in sync throughout.
function build_trajectory_separation_animation(
    long_trajectories_p1,
    reference_trajectory,
    growth_trajectories,
    t_axis,
    colors,
    group_labels,
    n_frames,
)
    sep_curves = [
        clamp.([norm(xn[:, k] .- reference_trajectory[:, k]) for k in 1:size(xn, 2)], 1e-12, Inf) for
        xn in growth_trajectories
    ]
    sep_ceil = 10^ceil(log10(maximum(reduce(vcat, sep_curves))))
    sep_ylims = (1e-3, sep_ceil)

    traj_xlims = extrema(vcat([xn[1, :] for xn in long_trajectories_p1]..., reference_trajectory[1, :]))
    traj_ylims = extrema(vcat([xn[2, :] for xn in long_trajectories_p1]..., reference_trajectory[2, :]))
    traj_zlims = extrema(vcat([xn[3, :] for xn in long_trajectories_p1]..., reference_trajectory[3, :]))

    frame_indices = round.(Int, range(1, size(reference_trajectory, 2), length = n_frames))

    return @animate for idx_g in frame_indices
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
            plot!(
                traj_panel,
                xn[1, 1:idx_g],
                xn[2, 1:idx_g],
                xn[3, 1:idx_g],
                color = colors[i],
                label = group_labels[i],
                linewidth = 2,
            )
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
end

# Plot 4/5: 2D loss surface |E(θ) - E_true| over (ρ, β).
function plot_loss_landscape(rho_range, beta_range, loss_grid, rho_true, beta_true)
    loss_plot = surface(
        rho_range,
        beta_range,
        loss_grid',
        xlabel = "ρ",
        ylabel = "β",
        zlabel = "loss",
        size = (800, 600),
        legend = false,
        guidefontsize = 24,
        tickfontsize = 18,
        grid = false,
    )
    scatter!(
        loss_plot,
        [rho_true],
        [beta_true],
        [0.0],
        color = :white,
        markershape = :star5,
        markersize = 16,
        markerstrokecolor = :black,
        label = false,
    )
    return loss_plot
end
