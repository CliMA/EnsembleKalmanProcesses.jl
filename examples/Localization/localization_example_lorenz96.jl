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
y = y0 + randn(D)
Γ = 1.0 * I

#### Define prior information on parameters
priors = map(1:p) do i
    constrained_gaussian(string("u", i), 0.0, 1.0, -Inf, Inf)
end
prior = combine_distributions(priors)

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ens)


# Solve problem without localization
ekiobj_vanilla =
    EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng, scheduler = DefaultScheduler(), localization_method = NoLocalization())
for i in 1:N_iter
    g_ens_vanilla = G(get_ϕ_final(prior, ekiobj_vanilla))
    EKP.update_ensemble!(ekiobj_vanilla, g_ens_vanilla, deterministic_forward_map = true)
end
nonlocalized_error = get_error(ekiobj_vanilla)[end]
@info "EKI - complete"
# Test Inflated
ekiobj_inflated = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    Inversion();
    rng = rng,
    scheduler = DefaultScheduler(),
    localization_method = NoLocalization(),
)

for i in 1:N_iter
    g_ens = G(get_ϕ_final(prior, ekiobj_inflated))
    EKP.update_ensemble!(
        ekiobj_inflated,
        g_ens,
        deterministic_forward_map = true,
        additive_inflation = true,
        additive_inflation_cov = 0.1 * I,
    )
end
#@info "EKI (Benoulli) - complete"
@info "EKI (inflated) - complete"

# Test SEC
ekiobj_sec = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    Inversion();
    rng = rng,
    localization_method = SEC(1.0),
    scheduler = DefaultScheduler(),
)

for i in 1:N_iter
    g_ens = G(get_ϕ_final(prior, ekiobj_sec))
    EKP.update_ensemble!(ekiobj_sec, g_ens, deterministic_forward_map = true)
end
@info "EKI (SEC) - complete"

# Test SEC with cutoff
ekiobj_sec_cutoff = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    Inversion();
    rng = rng,
    localization_method = SEC(1.0, 0.1),
    scheduler = DefaultScheduler(),
)

for i in 1:N_iter
    g_ens = G(get_ϕ_final(prior, ekiobj_sec_cutoff))
    EKP.update_ensemble!(ekiobj_sec_cutoff, g_ens, deterministic_forward_map = true)
end
@info "EKI (SEC cut-off) - complete"

# Test SECFisher
ekiobj_sec_fisher = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    Inversion();
    rng = rng,
    localization_method = SECFisher(),
    scheduler = DefaultScheduler(),
)

for i in 1:N_iter
    g_ens = G(get_ϕ_final(prior, ekiobj_sec_fisher))
    EKP.update_ensemble!(ekiobj_sec_fisher, g_ens, deterministic_forward_map = true)
end
@info "EKI (SEC Fisher) - complete"

# Test SECNice
ekiobj_sec_nice = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    Inversion();
    rng = rng,
    localization_method = SECNice(),
    scheduler = DefaultScheduler(),
)

for i in 1:N_iter
    g_ens = G(get_ϕ_final(prior, ekiobj_sec_nice))
    EKP.update_ensemble!(ekiobj_sec_nice, g_ens, deterministic_forward_map = true)
end
@info "EKI (SEC Nice) - complete"


u_final = get_u_final(ekiobj_sec)
g_final = get_g_final(ekiobj_sec)
cov_est = cov([u_final; g_final], [u_final; g_final], dims = 2, corrected = false)
# need dimension args too
cov_localized =
    get_localizer(ekiobj_sec).localize(cov_est, eltype(g_final), size(u_final, 1), size(g_final, 1), size(u_final, 2))

fig = plot(
    get_error(ekiobj_vanilla),
    label = "No localization",
    c = :Accent_6,
    lw = 6,
    size = (800 * 1.618, 800),
    xtickfont = 16,
    ytickfont = 16,
    guidefont = 16,
    legendfont = 16,
    bottom_margin = 5Plots.mm,
    left_margin = 5Plots.mm,
)
plot!(get_error(ekiobj_inflated), label = "Inflation only", lw = 6, ls=:dash)
plot!(get_error(ekiobj_sec), label = "SEC (Lee, 2021)", lw = 6)
plot!(get_error(ekiobj_sec_fisher), label = "SECFisher (Flowerdew, 2015)", lw = 6)
plot!(get_error(ekiobj_sec_cutoff), label = "SEC with cutoff", lw = 6)
plot!(get_error(ekiobj_sec_nice), label = "SEC NICE", lw = 6)


xlabel!("Iterations")
ylabel!("Error")
savefig(fig, "sec_comparison_lorenz96.png")
