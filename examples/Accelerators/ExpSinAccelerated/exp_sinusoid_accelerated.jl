
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

    cases = ["ens25-step1e-1", "ens10-step1e-1", "ens4-step1e-1"]
    case = cases[2]

    @info "running case $case"
    if case == "ens25-step1e-1"
        scheduler = DefaultScheduler(0.1)
        N_ens = 25
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "ens10-step1e-1"
        scheduler = DefaultScheduler(0.1) #DataMisfitController(terminate_at = 1e4)
        N_ens = 10
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "ens4-step1e-1"
        scheduler = DefaultScheduler(0.1) #DataMisfitController(terminate_at = 1e4)
        N_ens = 4
        localization_method = EKP.Localizers.NoLocalization() #.SEC(1.0, 0.01)
    end

    dim_output = 2
    Γ = 0.01 * I
    noise_dist = MvNormal(zeros(dim_output), Γ)
    theta_true = [1.0, 0.8]

    # We define a variety of prior distributions so we can study
    # the effectiveness of accelerators on this problem.

    prior_u1 = constrained_gaussian("amplitude", 2, 0.1, 0, 10)
    prior_u2 = constrained_gaussian("vert_shift", 0, 0.5, -10, 10)
    prior = combine_distributions([prior_u1, prior_u2])

    # To compare the two EKI methods, we will average over several trials, 
    # allowing the methods to run with different initial ensembles and noise samples.
    N_iterations = 100
    N_trials = 50
    @info "obtaining statistics over $N_trials trials"

    ## Solving the inverse problem

    # Preallocate so we can track and compare convergences of the methods
    all_convs = zeros(N_trials, N_iterations)
    all_convs_acc = zeros(N_trials, N_iterations)
    all_convs_acc_cs = zeros(N_trials, N_iterations)

    for trial in 1:N_trials
        # We now generate the initial ensemble and set up two EKI objects, one using an accelerator, 
        # to compare convergence.
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        ytrial = vec(G(theta_true) .+ rand(noise_dist))
        observation = Observation(Dict("samples" => ytrial, "covariances" => Γ, "names" => ["amplitude_vertshift"]))

        ensemble_kalman_process = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = DefaultAccelerator(),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = NesterovAccelerator(),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc_cs = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = FirstOrderNesterovAccelerator(),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        global mom_coeffs = zeros(N_iterations)
        global mom_coeffs_cs = zeros(N_iterations)

        # TODO SPLIT UP THIS LOOP !
        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.
        # We perform the inversion in parallel to compare the two EKI methods.
        for i in 1:N_iterations
            params_i = get_ϕ_final(prior, ensemble_kalman_process)
            params_i_acc = get_ϕ_final(prior, ensemble_kalman_process_acc)
            params_i_acc_cs = get_ϕ_final(prior, ensemble_kalman_process_acc_cs)

            G_ens = hcat([G(params_i[:, i]) for i in 1:N_ens]...)
            G_ens_acc = hcat([G(params_i_acc[:, i]) for i in 1:N_ens]...)
            G_ens_acc_cs = hcat([G(params_i_acc_cs[:, i]) for i in 1:N_ens]...)

            EKP.update_ensemble!(ensemble_kalman_process, G_ens, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc, G_ens_acc, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc_cs, G_ens_acc_cs, deterministic_forward_map = false)

            # save momentum coefficients
            mom_coeffs[i] = ensemble_kalman_process_acc.accelerator.θ_prev
            mom_coeffs_cs[i] = (
                1 -
                ensemble_kalman_process_acc_cs.accelerator.r / (get_N_iterations(ensemble_kalman_process_acc_cs) + 3)
            )
        end
        all_convs[trial, 1:length(get_error(ensemble_kalman_process))] = log.(get_error(ensemble_kalman_process))
        all_convs_acc[trial, 1:length(get_error(ensemble_kalman_process_acc))] =
            log.(get_error(ensemble_kalman_process_acc))
        all_convs_acc_cs[trial, 1:length(get_error(ensemble_kalman_process_acc_cs))] =
            log.(get_error(ensemble_kalman_process_acc_cs))
    end

    gr(size = (600, 500), legend = true)
    p = plot(
        1:N_iterations,
        mean(all_convs, dims = 1)[:],
        ribbon = std(all_convs, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = "",
        titlefont = 20,
        legendfontsize = 13,
        guidefontsize = 15,
        tickfontsize = 15,
        linewidth = 2,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc, dims = 1)[:],
        ribbon = std(all_convs_acc, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = "",
    )

    xlabel!("Iteration")
    ylabel!("log(Cost)")
    title!("EKI convergence on Exp Sin IP") #\n" * L"N_{ens} = " * "$N_ens; " * L"$\Delta t$ = 0.1")

    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.png"))
    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.pdf"))



end

main()
