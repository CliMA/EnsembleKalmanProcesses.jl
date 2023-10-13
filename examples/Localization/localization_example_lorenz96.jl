using Distributions
using LinearAlgebra
using Random

using Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers
import EnsembleKalmanProcesses: construct_mean, construct_cov, construct_sigma_ensemble
const EKP = EnsembleKalmanProcesses

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

D = 200

lorenz96_sys = (t, u) -> lorenz96(t, u, Dict("N" => D))

# Seed for pseudo-random number generator
rng_seed = 42
rng = Random.MersenneTwister(rng_seed)

dt = 0.05
y0 = rk4(lorenz96_sys, randn(D), 0.0, 1000.0, dt)

# Lorenz96 initial condition problem - Section 6.3 of Tong and Morzfeld (2022)
G(u) = mapslices((u) -> rk4(lorenz96_sys, u, 0.0, 0.4, dt), u, dims = 1)
N_ens = 20
p = D
N_iter = 20
# Generate random truth
Γ = 1.0 * I


n_repeats = 20

ekiobj_vanilla_err = zeros(n_repeats, N_iter)
ekiobj_bernoulli_err = zeros(n_repeats, N_iter)
ekiobj_sec_err = zeros(n_repeats, N_iter)
ekiobj_sec_fisher_err = zeros(n_repeats, N_iter)
ekiobj_cut_err = zeros(n_repeats, N_iter)
#ekiobj_cut2_err= zeros(n_repeats, N_iter)
ekiobj_sec_cutoff_err = zeros(n_repeats, N_iter)

#### Define prior information on parameters
prior = constrained_gaussian("u", 0.0, 1.0, -Inf, Inf, repeats = D)

for exp_it in 1:n_repeats
    # new data for each repeat
    y = y0 + randn(D)
    initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)

    # Solve problem without localization
    ekiobj_vanilla = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)
    for i in 1:N_iter
        g_ens_vanilla = G(get_ϕ_final(prior, ekiobj_vanilla))
        EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla, deterministic_forward_map = true)
    end
    nonlocalized_error = get_error(ekiobj_vanilla)[end]

    # Test Bernoulli
    ekiobj_bernoulli = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        y,
        Γ,
        Inversion();
        rng = rng,
        localization_method = BernoulliDropout(0.98),
    )

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_bernoulli))
        EKP.update_ensemble!(ekiobj_bernoulli, g_ens, deterministic_forward_map = true)
    end

    # Test SEC
    ekiobj_sec =
        EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = SEC(1.0))

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_sec))
        EKP.update_ensemble!(ekiobj_sec, g_ens, deterministic_forward_map = true)
    end

    # Test cutoff only
    ekiobj_cut =
        EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = SEC(0.0, 0.08))

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_cut))
        EKP.update_ensemble!(ekiobj_cut, g_ens, deterministic_forward_map = true)
    end

    # Test cutoff only
    #=
    ekiobj_cut2 = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = ThresholdCutoff(1.4))

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_cut2))
        EKP.update_ensemble!(ekiobj_cut2, g_ens, deterministic_forward_map = true)
    end
    =#
    # Test SEC & cutoff
    ekiobj_sec_cutoff =
        EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = SEC(1.0, 0.1))

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_sec_cutoff))
        EKP.update_ensemble!(ekiobj_sec_cutoff, g_ens, deterministic_forward_map = true)
    end

    # Test SECFisher
    ekiobj_sec_fisher =
        EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, localization_method = SECFisher())

    for i in 1:N_iter
        g_ens = G(get_ϕ_final(prior, ekiobj_sec_fisher))
        EKP.update_ensemble!(ekiobj_sec_fisher, g_ens, deterministic_forward_map = true)
    end

    ekiobj_vanilla_err[exp_it, :] = get_error(ekiobj_vanilla)
    ekiobj_bernoulli_err[exp_it, :] = get_error(ekiobj_bernoulli)
    ekiobj_sec_err[exp_it, :] = get_error(ekiobj_sec)
    ekiobj_sec_fisher_err[exp_it, :] = get_error(ekiobj_sec_fisher)
    ekiobj_cut_err[exp_it, :] = get_error(ekiobj_cut)
    #ekiobj_cut2_err[exp_it,:] = get_error(ekiobj_cut2)
    ekiobj_sec_cutoff_err[exp_it, :] = get_error(ekiobj_sec_cutoff)
end



fig = plot(mean(ekiobj_vanilla_err, dims = 1)[:], label = "No localization", size = (Int(floor(1.618 * 600)), 600))
plot!(mean(ekiobj_bernoulli_err, dims = 1)[:], label = "Bernoulli")
plot!(mean(ekiobj_sec_err, dims = 1)[:], label = "SEC (Lee, 2021)")
plot!(mean(ekiobj_sec_fisher_err, dims = 1)[:], label = "SECFisher (Flowerdew, 2015)")
plot!(mean(ekiobj_cut_err, dims = 1)[:], label = "(corr) cutoff")
#plot!(mean(ekiobj_cut2_err, dims=1)[:], label = "(cov) cutoff (Sans Alonso, 2023)")
plot!(mean(ekiobj_sec_cutoff_err, dims = 1)[:], label = "SEC with cutoff")

xlabel!("Iterations")
ylabel!("Error")
savefig(fig, "result.png")
