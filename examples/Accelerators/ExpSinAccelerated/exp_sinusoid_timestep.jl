using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LaTeXStrings
const EKP = EnsembleKalmanProcesses
fig_save_directory = joinpath(@__DIR__, "output/timestep")
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

    # cases = ["ens10-step1e-1", "ens10-step0.125", "ens10-step0.15"]
    # case = cases[3]

    # @info "running case $case"
    # if case == "ens10-step1e-1"
    #     timestep = 0.1
    #     scheduler = DefaultScheduler(timestep)
    #     N_ens = 10
    #     localization_method = EKP.Localizers.NoLocalization()
    # elseif case == "ens10-step0.125"
    #     timestep = 0.125
    #     scheduler = DefaultScheduler(timestep) #DataMisfitController(terminate_at = 1e4)
    #     N_ens = 10
    #     localization_method = EKP.Localizers.NoLocalization()
    # elseif case == "ens10-step0.15"
    #     timestep = 0.15
    #     scheduler = DefaultScheduler(timestep) #DataMisfitController(terminate_at = 1e4)
    #     N_ens = 10
    #     localization_method = EKP.Localizers.NoLocalization() #.SEC(1.0, 0.01)
    # end

    N_ens = 10
    localization_method = EKP.Localizers.NoLocalization()
    timestep_a = 0.75
    timestep_b = 0.5
    timestep_c = 0.25

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
    N_trials = 70
    @info "obtaining statistics over $N_trials trials"

    ## Solving the inverse problem

    # Preallocate so we can track and compare convergences of the methods
    all_convs_a = zeros(N_trials, N_iterations)
    all_convs_acc_a = zeros(N_trials, N_iterations)

    all_convs_b = zeros(N_trials, N_iterations)
    all_convs_acc_b = zeros(N_trials, N_iterations)

    all_convs_c = zeros(N_trials, N_iterations)
    all_convs_acc_c = zeros(N_trials, N_iterations)

    for trial in 1:N_trials
        # We now generate the initial ensemble and set up two EKI objects, one using an accelerator, 
        # to compare convergence.
        ytrial = vec(G(theta_true) .+ rand(noise_dist))
        observation = Observation(Dict("samples" => ytrial, "covariances" => Γ, "names" => ["amplitude_vertshift"]))


        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

        ensemble_kalman_process_a = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = DefaultAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_a),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc_a = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = NesterovAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_a),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_b = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = DefaultAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_b),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc_b = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = NesterovAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_b),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_c = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = DefaultAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_c),
            localization_method = deepcopy(localization_method),
        )
        ensemble_kalman_process_acc_c = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            observation,
            Inversion();
            accelerator = NesterovAccelerator(),
            rng = rng,
            scheduler = DefaultScheduler(timestep_c),
            localization_method = deepcopy(localization_method),
        )

        # We are now ready to carry out the inversion. At each iteration, we get the
        # ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
        # and apply the Kalman update to the ensemble.
        # We perform the inversion in parallel to compare the two EKI methods.

        # first, run the vanilla EKI objects for each timestep
        for i in 1:N_iterations
            params_i_a = get_ϕ_final(prior, ensemble_kalman_process_a)
            params_i_b = get_ϕ_final(prior, ensemble_kalman_process_b)
            params_i_c = get_ϕ_final(prior, ensemble_kalman_process_c)

            G_ens_a = hcat([G(params_i_a[:, i]) for i in 1:N_ens]...)
            G_ens_b = hcat([G(params_i_b[:, i]) for i in 1:N_ens]...)
            G_ens_c = hcat([G(params_i_c[:, i]) for i in 1:N_ens]...)

            EKP.update_ensemble!(ensemble_kalman_process_a, G_ens_a, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_b, G_ens_b, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_c, G_ens_c, deterministic_forward_map = false)
        end

        # now the Nesterov EKI objects
        for i in 1:N_iterations
            params_i_acc_a = get_ϕ_final(prior, ensemble_kalman_process_acc_a)
            params_i_acc_b = get_ϕ_final(prior, ensemble_kalman_process_acc_b)
            params_i_acc_c = get_ϕ_final(prior, ensemble_kalman_process_acc_c)

            G_ens_acc_a = hcat([G(params_i_acc_a[:, i]) for i in 1:N_ens]...)
            G_ens_acc_b = hcat([G(params_i_acc_b[:, i]) for i in 1:N_ens]...)
            G_ens_acc_c = hcat([G(params_i_acc_c[:, i]) for i in 1:N_ens]...)

            EKP.update_ensemble!(ensemble_kalman_process_acc_a, G_ens_acc_a, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc_b, G_ens_acc_b, deterministic_forward_map = false)
            EKP.update_ensemble!(ensemble_kalman_process_acc_c, G_ens_acc_c, deterministic_forward_map = false)
        end


        all_convs_a[trial, 1:length(get_error(ensemble_kalman_process_a))] = log.(get_error(ensemble_kalman_process_a))
        all_convs_acc_a[trial, 1:length(get_error(ensemble_kalman_process_acc_a))] =
            log.(get_error(ensemble_kalman_process_acc_a))
        all_convs_b[trial, 1:length(get_error(ensemble_kalman_process_b))] = log.(get_error(ensemble_kalman_process_b))
        all_convs_acc_b[trial, 1:length(get_error(ensemble_kalman_process_acc_b))] =
            log.(get_error(ensemble_kalman_process_acc_b))
        all_convs_c[trial, 1:length(get_error(ensemble_kalman_process_c))] = log.(get_error(ensemble_kalman_process_c))
        all_convs_acc_c[trial, 1:length(get_error(ensemble_kalman_process_acc_c))] =
            log.(get_error(ensemble_kalman_process_acc_c))
    end

    gr(size = (800, 600), legend = true)
    # plot ribbons
    p = plot(
        1:N_iterations,
        mean(all_convs_a, dims = 1)[:],
        ribbon = std(all_convs_a, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = L"\Delta t = " * "$timestep_a",
        titlefont = 20,
        legendfontsize = 13,
        guidefontsize = 15,
        tickfontsize = 15,
        linewidth = 3,
        fillalpha = 0.3,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_b, dims = 1)[:],
        ribbon = std(all_convs_b, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = L"\Delta t = " * "$timestep_b",
        linestyle = :dash,
        linewidth = 3,
        fillalpha = 0.3,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_c, dims = 1)[:],
        ribbon = std(all_convs_c, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = L"\Delta t = " * "$timestep_c",
        linestyle = :dot,
        linewidth = 3,
        fillalpha = 0.3,
    )

    plot!(
        1:N_iterations,
        mean(all_convs_acc_a, dims = 1)[:],
        ribbon = std(all_convs_acc_a, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = L"\Delta t = " * "$timestep_a, Nesterov",
        linewidth = 3,
        fillalpha = 0.3,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc_b, dims = 1)[:],
        ribbon = std(all_convs_acc_b, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = L"\Delta t = " * "$timestep_b, Nesterov",
        linestyle = :dash,
        linewidth = 3,
        fillalpha = 0.3,
    )
    plot!(
        1:N_iterations,
        mean(all_convs_acc_c, dims = 1)[:],
        ribbon = std(all_convs_acc_c, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = L"\Delta t = " * "$timestep_c, Nesterov",
        linestyle = :dot,
        linewidth = 3,
        fillalpha = 0.3,
    )

    # replot lines with alpha = 1
    xlabel!("Iteration")
    ylabel!("log(Cost)")
    title!("EKI convergence on Exp Sin IP") # \n" * L"N_{ens} = " * "$N_ens")

    savefig(p, joinpath(fig_save_directory, "timestep_comparison_exp_sin.pdf"))
    savefig(p, joinpath(fig_save_directory, "timestep_comparison_exp_sin.png"))

end

main()
