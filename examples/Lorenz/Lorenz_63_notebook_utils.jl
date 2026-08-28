# Supporting functions for Lorenz_63_notebook.jl: filesystem/preliminaries, data construction,
# and plotting. Assumes GModel_L63.jl and the required packages are already loaded by the caller.

###
### Preliminaries / filesystem
###

function make_output_dirs(root_dir::AbstractString = pwd(); figure_subdir::AbstractString = "output", data_subdir::AbstractString = "output")
    println(root_dir)
    figure_save_directory = joinpath(root_dir, figure_subdir)
    data_save_directory = joinpath(root_dir, data_subdir)
    isdir(figure_save_directory) || mkdir(figure_save_directory)
    isdir(data_save_directory) || mkdir(data_save_directory)
    return figure_save_directory, data_save_directory
end

# Spin up a random initial condition onto the attractor, ready for use as `x0`.
function spin_up_attractor_ic(rng_seed_init, nx, true_params_config, dt, T_spinup)
    rng_i = MersenneTwister(rng_seed_init)
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
    x_spun_up = lorenz_solve(true_params_config, x_initial, LorenzConfig(dt, T_spinup))
    x0 = x_spun_up[:, end]
    return x_initial, x0
end

###
### Data construction
###

function build_priors(param_names, prior_means, prior_stds, prior_bounds)
    marginals = [
        constrained_gaussian(param_names[i], prior_means[i], prior_stds[i], prior_bounds[i][1], prior_bounds[i][2]) for
        i in 1:length(param_names)
    ]
    return combine_distributions(marginals)
end

# Generates the (artificial) truth sample `y` and its internal-variability covariance `Γy`,
# estimated from `multiple` independent statistics windows of a long trajectory at the true
# parameters, and packages them into an `Observation`.
function generate_truth_and_covariance(true_params_config, x_initial, x0, dt, T, T_start, T_end, multiple, data_names)
    lorenz_config_settings = LorenzConfig(dt, T)
    observation_config = ObservationConfig(T_start, T_end)

    y = lorenz_forward(true_params_config, x0, lorenz_config_settings, observation_config)
    ny = length(y)

    println("Using truth values to compute covariance")
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

    truth = Observation(Dict("samples" => y, "covariances" => Γy, "names" => data_names))
    return lorenz_config_settings, observation_config, y, ny, Γy, truth
end

# Builds the forward map θ = (log ρ, log β) -> G(θ) used by Levenberg-Marquardt.
function build_lm_forward_map(sigma_true, x0, lorenz_config_settings, observation_config)
    return θ -> lorenz_forward(
        EnsembleMemberConfig(promote(sigma_true, exp(θ[1]), exp(θ[2]))...),
        x0,
        lorenz_config_settings,
        observation_config,
    )
end

# Levenberg-Marquardt (derivative-based) comparison, with exact Jacobians via ForwardDiff
# through the full chaotic Lorenz solve. Loss normalized as `1/ny * (y - G(θ))' * Γy⁻¹ * (y - G(θ))`,
# matching EKI's `get_error` (`compute_loss_at_mean`).
function run_levenberg_marquardt(θ_lm, N_iter, nu, R_inv_var, y, G_lm; λ0 = 1.0)
    λ = λ0
    ny_lm = length(y)
    lm_history = [exp.(θ_lm)]
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

###
### Plotting
###

function plot_error_convergence(err, lm_err)
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
        size = (800, 500),
        grid = false,
        guidefontsize = 24,
        tickfontsize = 18,
        legendfontsize = 21,
        left_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
    )
    plot!(err_plot, 1:length(lm_err), lm_err, linewidth = 4, marker = :diamond, color = :magenta, label = "LM")
    return err_plot
end

function plot_final_parameters(param_names, final_mean, params_true)
    param_plot_height = round(Int, 800 * 400 / (450 * length(param_names)))
    param_plot = plot(layout = (1, length(param_names)), size = (800, param_plot_height), legend = false)
    for (pp, pname) in enumerate(param_names)
        bar!(param_plot[pp], ["EKI"], [final_mean[pp]], title = pname)
        hline!(param_plot[pp], [params_true[pp]], linestyle = :dash, linecolor = :red)
    end
    return param_plot
end

function plot_eki_convergence(priors, ekiobj, n_eki_iterations, lm_history, params_true; eki_color = :black, lm_color = :magenta)
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
        size = (800, 622),
        guidefontsize = 24,
        tickfontsize = 18,
        grid = false,
        left_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
    )
    scatter!(convergence_plot, ϕ_final[1, :], ϕ_final[2, :], color = eki_color, alpha = 1.0, markersize = 6, markerstrokewidth = 0, label = false)
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
    return convergence_plot
end

# Illustrates the statistics window: the full [0, T] trajectory at the true parameters, in
# stacked x/y/z panels, with the [T_start, T_end] window used to compute calibration
# statistics marked by dashed red lines.
function plot_statistics_window(true_params_config, x0, dt, T, T_start, T_end)
    traj = lorenz_solve(true_params_config, x0, LorenzConfig(dt, T))
    t = (0:(size(traj, 2) - 1)) .* dt
    axis_labels = ["x", "y", "z"]

    panels = [
        plot(
            t,
            traj[i, :],
            color = :black,
            linewidth = 1.5,
            xlabel = i == 3 ? "Time" : "",
            ylabel = axis_labels[i],
            title = i == 1 ? "Statistics window [T_start, T_end]" : "",
            legend = false,
            grid = false,
            guidefontsize = 14,
            titlefontsize = 16,
            tickfontsize = 10,
        ) for i in 1:3
    ]
    for p in panels
        vline!(p, [T_start, T_end], linestyle = :dash, linecolor = :red, linewidth = 2)
    end
    return plot(panels..., layout = (3, 1), size = (800, 667))
end
