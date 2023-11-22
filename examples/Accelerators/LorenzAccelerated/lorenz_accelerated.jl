using Distributions
using LinearAlgebra
using Random

using Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses

fig_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end

function rk4(f::F, y0::Array{Float64, 1}, t0::Float64, t1::Float64, h::Float64; inplace::Bool = true) where {F}
    y = y0
    n = round(Int, (t1 - t0) / h)
    t = t0
    if ~inplace
        hist = zeros(n, length(y0))
    end
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if ~inplace
            hist[i, :] = y
        end
        t = t0 + i * h
    end
    if ~inplace
        return hist
    else
        return y
    end
end

function lorenz96(t, u, p)
    N = p["N"]
    F = 8

    du = similar(u)

    for i in 1:N
        du[i] = (u[mod(i + 1, 1:N)] - u[mod(i - 2, 1:N)]) * u[mod(i - 1, 1:N)] - u[i] + F
    end

    return copy(du)
end


function main()
    D = 20

    cases = ["const", "dmc", "dmc-loc-small-ens"]
    case = cases[3]

    @info "running case $case"
    if case == "const"
        scheduler = DefaultScheduler()
        N_ens = 20
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "dmc"
        scheduler = DataMisfitController(terminate_at = 1e4)
        N_ens = 20
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "dmc-loc-small-ens"
        scheduler = DataMisfitController(terminate_at = 1e4)
        N_ens = 5
        localization_method = EKP.Localizers.SEC(1.0, 0.01)
    end

    lorenz96_sys = (t, u) -> lorenz96(t, u, Dict("N" => D))

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)
    dt = 0.05
    y0 = rk4(lorenz96_sys, randn(D), 0.0, 1000.0, dt)

    # Lorenz96 initial condition problem - Section 6.3 of Tong and Morzfeld (2022)
    G(u) = mapslices((u) -> rk4(lorenz96_sys, u, 0.0, 0.4, dt), u, dims = 1)
    p = D
    # Generate random truth
    y = y0 + randn(D)
    Γ = 1.0 * I

    #### Define prior information on parameters
    priors = map(1:p) do i
        constrained_gaussian(string("u", i), 0.0, 1.0, -Inf, Inf)
    end
    prior = combine_distributions(priors)


    N_iter = 20
    N_trials = 50
    @info "obtaining statistics over $N_trials trials"

    errs = zeros(N_trials, N_iter)
    errs_acc = zeros(N_trials, N_iter)
    errs_acc_cs = zeros(N_trials, N_iter)

    for trial in 1:N_trials
        initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)


        # We create 3 EKP Inversion objects to compare acceleration.
        ekiobj_vanilla = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y,
            Γ,
            Inversion();
            rng = rng,
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_acc = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y,
            Γ,
            Inversion();
            rng = rng,
            accelerator = NesterovAccelerator(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_acc_cs = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            y,
            Γ,
            Inversion();
            rng = rng,
            accelerator = FirstOrderNesterovAccelerator(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        err = zeros(N_iter)
        err_acc = zeros(N_iter)
        err_acc_cs = zeros(N_iter)
        for i in 1:N_iter
            g_ens_vanilla = G(get_ϕ_final(prior, ekiobj_vanilla))
            EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla, deterministic_forward_map = true)
            g_ens_acc = G(get_ϕ_final(prior, ekiobj_acc))
            EKP.update_ensemble!(ekiobj_acc, g_ens_acc, deterministic_forward_map = true)
            g_ens_acc_cs = G(get_ϕ_final(prior, ekiobj_acc_cs))
            EKP.update_ensemble!(ekiobj_acc_cs, g_ens_acc_cs, deterministic_forward_map = true)
        end
        errs[trial, :] = get_error(ekiobj_vanilla)
        errs_acc[trial, :] = get_error(ekiobj_acc)
        errs_acc_cs[trial, :] = get_error(ekiobj_acc_cs)
    end

    # COMPARE CONVERGENCES

    convplot = plot(1:(N_iter), mean(errs, dims = 1)[:], color = :black, label = "No acceleration")
    plot!(1:(N_iter), mean(errs_acc, dims = 1)[:], color = :blue, label = "Nesterov")
    plot!(1:(N_iter), mean(errs_acc_cs, dims = 1)[:], color = :red, label = "Nesterov, FirstOrder")
    title!("EKI convergence on Lorenz96 IP, N_trials=" * string(N_trials))
    xlabel!("Iteration")

    # error bars
    plot!(
        1:(N_iter),
        (mean(errs, dims = 1)[:] + std(errs, dims = 1)[:] / sqrt(N_trials)),
        color = :black,
        ls = :dash,
        label = "",
    )
    plot!(
        1:(N_iter),
        (mean(errs, dims = 1)[:] - std(errs, dims = 1)[:] / sqrt(N_trials)),
        color = :black,
        ls = :dash,
        label = "",
    )
    plot!(
        1:(N_iter),
        (mean(errs_acc, dims = 1)[:] + std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
        color = :blue,
        ls = :dash,
        label = "",
    )
    plot!(
        1:(N_iter),
        (mean(errs_acc, dims = 1)[:] - std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
        color = :blue,
        ls = :dash,
        label = "",
    )
    plot!(
        1:(N_iter),
        (mean(errs_acc_cs, dims = 1)[:] + std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
        color = :red,
        ls = :dash,
        label = "",
    )
    plot!(
        1:(N_iter),
        (mean(errs_acc_cs, dims = 1)[:] - std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
        color = :red,
        ls = :dash,
        label = "",
    )

    savefig(convplot, joinpath(fig_save_directory, case * "_lorenz96.png"))
end

main()
