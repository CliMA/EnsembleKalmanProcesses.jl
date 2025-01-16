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

    cases = ["momcoeffs_ens25-step1e-1-false", "momcoeffs_ens10-step1e-1-false", "momcoeffs_ens4-step1e-1-false"]
    case = cases[1]

    @info "running case $case"
    if case == "momcoeffs_ens25-step1e-1-false"
        scheduler = DefaultScheduler(0.1)
        N_ens = 25
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "momcoeffs_ens10-step1e-1-false"
        scheduler = DefaultScheduler(0.1) #DataMisfitController(terminate_at = 1e4)
        N_ens = 10
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "momcoeffs_ens4-step1e-1-false"
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
    N_iterations = 500
    N_trials = 50
    @info "obtaining statistics over $N_trials trials"

    ## Solving the inverse problem

    # Preallocate so we can track and compare convergences of the methods
    all_convs = zeros(N_trials, N_iterations)
    all_convs_acc = zeros(N_trials, N_iterations)
    all_convs_acc_cs = zeros(N_trials, N_iterations)
    all_convs_acc_const = zeros(N_trials, N_iterations)

    for trial in 1:N_trials
        ytrial = vec(G(theta_true) .+ rand(noise_dist))
        observation = Observation(Dict("samples" => ytrial, "covariances" => Γ, "names" => ["amplitude_vertshift"]))

        # We now generate the initial ensemble and set up two EKI objects, one using an accelerator, 
        # to compare convergence.
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        ensemble_kalman_process = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            rng = rng,
            accelerator = DefaultAccelerator(),
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
        ensemble_kalman_process_acc_const9 = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = ConstantNesterovAccelerator(0.9),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc_const5 = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = ConstantNesterovAccelerator(0.5),
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        global mom_coeffs = zeros(N_iterations)
        global mom_coeffs_cs = zeros(N_iterations)

        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.

        for i in 1:N_iterations  # vanilla EKI
            params_i = get_ϕ_final(prior, ensemble_kalman_process)
            G_ens = hcat([G(params_i[:, i]) for i in 1:N_ens]...)
            EKP.update_ensemble!(ensemble_kalman_process, G_ens, deterministic_forward_map = false)
        end

        for i in 1:N_iterations # NesterovAccelerator
            params_i_acc = get_ϕ_final(prior, ensemble_kalman_process_acc)
            G_ens_acc = hcat([G(params_i_acc[:, i]) for i in 1:N_ens]...)
            EKP.update_ensemble!(ensemble_kalman_process_acc, G_ens_acc, deterministic_forward_map = false)
            # save momentum coefficients
            b = ensemble_kalman_process_acc.accelerator.θ_prev^2
            θ = (-b + sqrt(b^2 + 4 * b)) / 2
            mom_coeffs[i] = θ * (1 / ensemble_kalman_process_acc.accelerator.θ_prev - 1)
        end

        for i in 1:N_iterations #FirstOrderNesterovAccelerator
            params_i_acc_cs = get_ϕ_final(prior, ensemble_kalman_process_acc_cs)
            G_ens_acc_cs = hcat([G(params_i_acc_cs[:, i]) for i in 1:N_ens]...)
            EKP.update_ensemble!(ensemble_kalman_process_acc_cs, G_ens_acc_cs, deterministic_forward_map = false)
            # save momentum coefficients
            mom_coeffs_cs[i] = (
                1 -
                ensemble_kalman_process_acc_cs.accelerator.r / (get_N_iterations(ensemble_kalman_process_acc_cs) + 3)
            )
        end

        for i in 1:N_iterations  # constant, lambda=0.9
            params_i_acc_const = get_ϕ_final(prior, ensemble_kalman_process_acc_const9)
            G_ens_acc_const = hcat([G(params_i_acc_const[:, i]) for i in 1:N_ens]...)
            EKP.update_ensemble!(ensemble_kalman_process_acc_const9, G_ens_acc_const, deterministic_forward_map = false)
        end
        all_convs[trial, 1:length(get_error(ensemble_kalman_process))] = log.(get_error(ensemble_kalman_process))
        all_convs_acc[trial, 1:length(get_error(ensemble_kalman_process_acc))] =
            log.(get_error(ensemble_kalman_process_acc))
        all_convs_acc_cs[trial, 1:length(get_error(ensemble_kalman_process_acc_cs))] =
            log.(get_error(ensemble_kalman_process_acc_cs))
        all_convs_acc_const[trial, 1:length(get_error(ensemble_kalman_process_acc_const9))] =
            log.(get_error(ensemble_kalman_process_acc_const9))
    end

    gr(size = (700, 600), legend = true)

    p = plot(
        1:N_iterations,
        mean(all_convs, dims = 1)[:],
        ribbon = std(all_convs, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = "No acceleration",
        titlefont = 24,
        legendfontsize = 18,
        guidefontsize = 20,
        tickfontsize = 20,
        linewidth = 2,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc_cs, dims = 1)[:],
        ribbon = std(all_convs_acc_cs, dims = 1)[:] / sqrt(N_trials),
        color = :red,
        label = "Original coefficient",
        linewidth = 2,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc, dims = 1)[:],
        ribbon = std(all_convs_acc, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = "Recursive coefficient",
        linewidth = 2,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc_const, dims = 1)[:],
        ribbon = std(all_convs_acc_const, dims = 1)[:] / sqrt(N_trials),
        color = :green,
        label = "Constant coefficient",
        linewidth = 2,
    )
    xlabel!("Iteration")
    ylabel!("log(Cost)")
    title!("EKI convergence on Exp Sin IP")
    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.png"))
    savefig(p, joinpath(fig_save_directory, case * "_exp_sin.pdf"))

    plot_x = 1:40
    coeff_plot = plot(plot_x, mom_coeffs_cs[plot_x], color = :red, label = "Original coefficient", linewidth = 2)
    plot!(
        plot_x,
        mom_coeffs[plot_x],
        color = :blue,
        label = "Recursive coefficient",
        titlefont = 24,
        legendfontsize = 18,
        guidefontsize = 20,
        tickfontsize = 20,
        linewidth = 2,
    )
    plot!(plot_x, ones(length(plot_x)) * 0.9, color = :green, label = "Constant coefficient", linewidth = 2)
    title!("Momentum coefficient values")
    xlabel!("Iteration")
    savefig(coeff_plot, joinpath(fig_save_directory, "coeff_evolution_exp_sin.png"))
    savefig(coeff_plot, joinpath(fig_save_directory, "coeff_evolution_exp_sin.pdf"))
end

main()
