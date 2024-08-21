# # Comparison of different Learning rate schedulers

# In this example we have a model that produces the exponential of a sinusoid
# ``f(A, v) = exp(A \sin(t) + v), \forall t \in [0,2\pi]``. Given an initial guess of the parameters as
# ``A^* \sim \mathcal{N}(2,1)`` and ``v^* \sim \mathcal{N}(0,5)``, our goal is
# to estimate the parameters from a noisy observation of the maximum, minimum,
# and mean of the true model output.

# We repeat the experiment using several timestepping methods with EKI,
# we also repeat the experiment over many seeds

# We produce 3 plots:
# (1) Final model ensembles of a single run, and reproduction of the observed data (max & mean of model)
# (2) Convergence of a single run over iterations, and the average performance over many runs.
# (3) Convergence of a single run over algorithm time, and the average performance ovver many runs.


# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots, ColorSchemes
using Plots.PlotMeasures
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

## We set up the file syste,

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

# Seed for pseudo-random number generator.
repeats = 100
rng_seeds = [199 + 135 * i for i in 1:repeats]
rngs = [Random.MersenneTwister(rs) for rs in rng_seeds]

# Now, we define a model which generates a sinusoid given parameters ``\theta``: an
# amplitude and a vertical shift. It then is exponentiated, We will estimate these parameters from data.

# The model adds a random phase shift upon evaluation.
dt = 0.01
trange = 0:dt:(2 * pi + dt)
function model(amplitude, vert_shift)
    return exp.(amplitude * sin.(trange) .+ vert_shift)
end

# We then define ``G(\theta)``, which returns the observables of the sinusoid
# given a parameter vector. These observables should be defined such that they
# are informative about the parameters we wish to estimate. 
function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve), mean(sincurve)]
end

# Suppose we have a noisy observation of the true system. Here, we create a
# pseudo-observation ``y`` by running our model with the correct parameters
# and adding Gaussian noise to the output.
dim_output = 2

Γ = 0.1 * I
noise_dist = MvNormal(zeros(dim_output), Γ)

u_true = [1.0, 0.8]
y_nonoise = G(u_true)
y = y_nonoise .+ rand(rngs[1], noise_dist)

gr()

## Solving the inverse problem

# We now define prior distributions on the two parameters. For the amplitude,
# we define a prior with mean 2 and standard deviation 1. It is
# additionally constrained to be nonnegative. For the vertical shift we define
# a Gaussian prior with mean 0 and standard deviation 5.
prior_u1 = constrained_gaussian("amplitude", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("vert_shift", 0, 5, -Inf, Inf)
prior = combine_distributions([prior_u1, prior_u2])
unconstrained_u_true = transform_constrained_to_unconstrained(prior, u_true)

# We now generate the initial ensemble and set up the ensemble Kalman inversion.
N_ensemble = 50
N_iterations = 50

# Set up some initial plot information
clrs = palette(:tab10)
plt = plot(
    trange,
    model(u_true...),
    c = :black,
    label = "Truth",
    legend = :topright,
    linewidth = 2,
    title = "Solution evaluation",
)
xlabel!(plt, "Time")
plt_thin = plot(grid = false, xticks = false, title = "(max, mean)")
for i in 1:2
    plot!(
        plt_thin,
        trange,
        repeat([y_nonoise[i]], length(trange)),
        ribbon = [2 * sqrt(Γ[i, i]); 2 * sqrt(Γ[i, i])],
        linestyle = :dash,
        c = :grey,
        label = "",
    )
    plot!(plt_thin, trange, repeat([y[i]], length(trange)), c = :black, label = "")
end
plt2 =
    plot(xlabel = "iterations", yscale = :log10, ylim = [1e-4, 1e2], ylabel = "L2 norm", title = "Example convergence")
plt3 = plot(
    xlabel = "algorithm time",
    yscale = :log10,
    ylim = [1e-4, 1e2],
    ylabel = "L2 norm",
    title = "Example convergence",
)


final_misfit = zeros(5, repeats)
final_u_err = zeros(5, repeats)
final_u_spread = zeros(5, repeats)
ts_tmp = []

# We run two loops, over `repeats` number of random initial samples
# and over the collection of schedulers `timestepppers`

for (rng_idx, rng) in enumerate(rngs)
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)
    N_iters = repeat([N_iterations], 5)
    schedulers = [
        DefaultScheduler(0.5),
        DefaultScheduler(0.02),
        EKSStableScheduler(),
        DataMisfitController(),
        DataMisfitController(on_terminate = "continue"),
    ]
    push!(ts_tmp, schedulers)
    for (idx, scheduler, N_iter) in zip(1:length(schedulers), schedulers, N_iters)
        ensemble_kalman_process =
            EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); scheduler = scheduler, rng = rng)

        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.

        u_err = zeros(N_iter + 1)
        u_spread = zeros(N_iter + 1)
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ensemble_kalman_process)
            u_i = get_u_final(ensemble_kalman_process)
            u_err[i] = 1 / size(u_i, 2) * sum((u_i .- unconstrained_u_true) .^ 2)
            u_spread[i] = 1 / size(u_i, 2) * sum((u_i .- mean(u_i, dims = 2)) .^ 2)
            G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

            terminated = EKP.update_ensemble!(ensemble_kalman_process, G_ens)
            if !isnothing(terminated)
                N_iters[idx] = i - 1
                break # if the timestep was terminated due to timestepping condition
            end

        end
        # this will change in on failure condition
        N_iter = N_iters[idx]

        # required for plots
        u_err = u_err[1:(N_iter + 1)]
        u_spread = u_spread[1:(N_iter + 1)]
        u_i = get_u_final(ensemble_kalman_process)
        u_err[N_iter + 1] = 1 / size(u_i, 2) * sum((u_i .- unconstrained_u_true) .^ 2)
        u_spread[N_iter + 1] = 1 / size(u_i, 2) * sum((u_i .- mean(u_i, dims = 2)) .^ 2)

        final_u_err[idx, rng_idx] = u_err[end]
        final_u_spread[idx, rng_idx] = u_spread[end]

        Δt = get_Δt(ensemble_kalman_process)
        alg_time = [sum(Δt[1:i]) for i in 1:length(Δt)]
        pushfirst!(alg_time, 0.0)
        misfit = get_error(ensemble_kalman_process)
        final_misfit[idx, rng_idx] = misfit[end]

        if rng_idx == 1
            plot!(plt2, 0:N_iter, u_err, c = clrs[idx + 1], label = "$(nameof(typeof(scheduler)))")
            plot!(plt2, 0:N_iter, u_spread, c = clrs[idx + 1], ls = :dash, label = "")

            plot!(plt3, alg_time, u_err, c = clrs[idx + 1], label = "$(nameof(typeof(scheduler)))")
            plot!(plt3, alg_time, u_spread, c = clrs[idx + 1], ls = :dash, label = "")

            # Finally, we get the ensemble after the last iteration. This provides our estimate of the parameters.
            final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

            # To visualize the success of the inversion, we plot model with the true
            # parameters, the initial ensemble, and the final ensemble.


            plot!(
                plt,
                trange,
                model(mean(final_ensemble, dims = 2)...),
                alpha = 1.0,
                c = clrs[idx + 1],
                label = "$(nameof(typeof(scheduler)))",
            )
            G_final_mean = G(mean(final_ensemble, dims = 2)[:])
            for k in 1:2
                plot!(
                    plt_thin,
                    trange,
                    repeat([G_final_mean[k]], length(trange)),
                    linestyle = :dot,
                    c = clrs[idx + 1],
                    label = "",
                )
            end
        end
    end
end

#some final plot tweaking  / combining
ylims!(plt_thin, ylims(plt))
ll = @layout [a{0.8w} b{0.2w}]
plt = plot(plt, plt_thin, layout = ll, right_margin = 10mm)

mean_final_misfit = mean(final_misfit, dims = 2)
mean_final_u_err = mean(final_u_err, dims = 2)
mean_final_u_spread = mean(final_u_spread, dims = 2)


plot!(plt2, [1], [ylims(plt2)[2] + 1], c = :gray, label = "error")
plot!(plt2, [1], [ylims(plt2)[2] + 1], c = :gray, ls = :dash, label = "spread")
plot!(plt3, [1], [ylims(plt3)[2] + 1], c = :gray, label = "error")
plot!(plt3, [1], [ylims(plt3)[2] + 1], c = :gray, ls = :dash, label = "spread")
plt2_thin = plot(yscale = :log10, grid = false, xticks = false, title = "Final ($(repeats) runs)")
for i in 1:5
    plot!(
        plt2_thin,
        trange,
        repeat([mean_final_u_err[i]], length(trange)),
        yscale = :log10,
        c = clrs[i + 1], #idx + 1 for colours
        label = "",
    )
    plot!(
        plt2_thin,
        trange,
        repeat([mean_final_u_spread[i]], length(trange)),
        c = clrs[i + 1], #idx + 1 for colours
        ls = :dash,
        label = "",
    )
end
plt3_thin = plt2_thin
ylims!(plt2_thin, ylims(plt2))
plt2 = plot(plt2, plt2_thin, layout = ll, right_margin = 10mm)
ylims!(plt3_thin, ylims(plt3))
plt3 = plot(plt3, plt3_thin, layout = ll, right_margin = 10mm)

for (i, t) in enumerate(ts_tmp[1])
    println(" ")
    println("Method      : ", t)
    println("Final misfit: ", mean_final_misfit[i])
    println("Final error : ", mean_final_u_err[i])
    println("Final spread: ", mean_final_u_spread[i])
end

savefig(plt, joinpath(figure_save_directory, "ensembles.png"))
savefig(plt2, joinpath(figure_save_directory, "error_vs_spread_over_iteration.png"))
savefig(plt3, joinpath(figure_save_directory, "error_vs_spread_over_time.png"))
savefig(plt, joinpath(figure_save_directory, "ensembles.pdf"))
savefig(plt2, joinpath(figure_save_directory, "error_vs_spread_over_iteration.pdf"))
savefig(plt3, joinpath(figure_save_directory, "error_vs_spread_over_time.pdf"))
