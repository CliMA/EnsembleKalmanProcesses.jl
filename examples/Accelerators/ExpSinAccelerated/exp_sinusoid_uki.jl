
# This example is a modification of the sinusoid example problem; the exponential
# makes the inverse problem highly nonlinear, which makes it a suitable problem on 
# which to test acceleration methods.

# In this example we have a model that produces an exponential sinusoid function
# ``f(A, v) = \exp(A \sin(\phi + t) + v), \forall t \in [0,2\pi]``, with a random
# phase ``\phi``. Given an initial guess of the parameters using some multivariate
# normal distribution, our goal is to estimate the parameters from a noisy observation 
# of the maximum, minimum, and mean of the true model output. We will compare the
# parameter estimates achieved through traditional EKI and through accelerated versions
# of the EKI algorithm.

# First, we load the packages we need:
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LaTeXStrings

const EKP = EnsembleKalmanProcesses
fig_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end

## Setting up the model and data for our inverse problem
dt = 0.01
trange = 0:dt:(2 * pi + dt)
function model(amplitude, vert_shift)
    phi = 2 * pi * rand(rng)
    return exp.(amplitude * sin.(trange .+ phi) .+ vert_shift)
end
nothing # hide

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
nothing # hide

function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end

function main()

    cases = ["uki_5e-2"]
    case = cases[1]

    @info "running case $case"
    if case == "uki_5e-2"
        scheduler = DefaultScheduler(0.05)
        localization_method = EKP.Localizers.NoLocalization()
    end

    dim_output = 2
    Γ = 0.01 * I
    noise_dist = MvNormal(zeros(dim_output), Γ)
    theta_true = [1.0, 0.8]

    prior_u1 = constrained_gaussian("amplitude", 2, 0.1, 0, 10)
    prior_u2 = constrained_gaussian("vertshift", 0, 0.5, -10, 10)
    prior = combine_distributions([prior_u1, prior_u2])

    N_iterations = 100
    N_trials = 100
    @info "obtaining statistics over $N_trials trials"
    # Define cost function to compare convergences. We use a logarithmic cost function 
    # to best interpret exponential model. Note we do not explicitly penalize distance from the prior here.
    function cost(theta, y)
        return log.(norm(inv(Γ) .^ 0.5 * (G(theta) .- y)) .^ 2)
    end

    ## Solving the inverse problem

    # Preallocate so we can track and compare convergences of the methods
    all_convs = zeros(N_trials, N_iterations)
    all_convs_acc = zeros(N_trials, N_iterations)

    for trial in 1:N_trials
        ytrial = vec(G(theta_true) .+ rand(noise_dist))
        # a different observational noise sample drawn for each trial provides variety; initial ensembles are deterministic in UKI
        observation = Observation(Dict("samples" => ytrial, "covariances" => Γ, "names" => ["amplitude_vertshift"]))

        priormean = mean(prior)
        priorcov = cov(prior)

        ensemble_kalman_process = EKP.EnsembleKalmanProcess(
            observation,
            Unscented(priormean, priorcov);
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc = EKP.EnsembleKalmanProcess(
            observation,
            Unscented(priormean, priorcov);
            accelerator = NesterovAccelerator(),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        global convs = zeros(N_iterations)
        global convs_acc = zeros(N_iterations)

        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.

        for i in 1:N_iterations  # vanilla UKI
            params_i = get_ϕ_final(prior, ensemble_kalman_process)
            G_ens = hcat([G(params_i[:, i]) for i in 1:size(params_i)[2]]...)
            EKP.update_ensemble!(ensemble_kalman_process, G_ens)
            convs[i] = cost(mean(params_i, dims = 2), ytrial)
        end

        for i in 1:N_iterations # NesterovAccelerator
            params_i_acc = get_ϕ_final(prior, ensemble_kalman_process_acc)
            G_ens_acc = hcat([G(params_i_acc[:, i]) for i in 1:size(params_i_acc)[2]]...)
            EKP.update_ensemble!(ensemble_kalman_process_acc, G_ens_acc)
            convs_acc[i] = cost(mean(params_i_acc, dims = 2), ytrial)
        end

        all_convs[trial, :] = convs
        all_convs_acc[trial, :] = convs_acc
    end

    gr(size = (800, 600), legend = true)
    # p = plot(1:N_iterations, mean(all_convs, dims = 1)[:], color = :black, label = "No acceleration", titlefont=20, legendfontsize=13,guidefontsize=15,tickfontsize=15, linewidth=2)
    # plot!(1:N_iterations, mean(all_convs_acc, dims = 1)[:], color = :blue, label = "Nesterov",)
    # error bars
    p = plot(
        1:N_iterations,
        mean(all_convs, dims = 1)[:],
        ribbon = std(all_convs, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = "No acceleration",
        titlefont = 23,
        legendfontsize = 16,
        guidefontsize = 18,
        tickfontsize = 18,
        linewidth = 2,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc, dims = 1)[:],
        ribbon = std(all_convs_acc, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = "Nesterov",
        linewidth = 2,
    )
    xlabel!("Iteration")
    ylabel!("log(Cost)")
    title!("UKI convergence on Exp Sin IP")

    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.png"))
    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.pdf"))
end

main()
