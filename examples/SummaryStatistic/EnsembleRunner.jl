# Shared utilities for building ensembles of driving trajectories u(t) = μ(t) (the exploratory
# climate forcing), used by `generate_training_data.jl` (μ never crosses the bistability
# threshold) and `generate_testing_data.jl` (μ is driven across it, forcing a tipping event).
using Random
using Statistics

include("L96MultiscaleModel.jl")
include("SummaryStatistics.jl")

# Saddle-node threshold: for a >= 0, the bistable well c^3 - a*c - μ = 0 has three real roots
# iff |μ| < μ_critical(a) = 2/(3√3) * a^{3/2}.
μ_critical(a::Real) = 2 / (3 * sqrt(3)) * a^(3 / 2)

# Draw (μ0, r) such that μ(t) = μ0 + r*t stays within `safety * μ_c` of zero for every
# t in [0, T]. Constructed directly (not by rejection): given μ0, the largest r magnitude that
# cannot reach the bound by time T is (bound - |μ0|)/T, so sampling r within that range
# guarantees |μ0 + r*t| <= bound for all t in [0, T] (μ is monotone in t).
function sample_nontipping_mu(rng::AbstractRNG, T::Real, a::Real; μ_bound::Real = 0.3, safety::Real = 0.6)
    bound = safety * μ_critical(a)
    μ0 = min(μ_bound, bound) * (2 * rand(rng) - 1)
    r_max_allowed = (bound - abs(μ0)) / T
    r = r_max_allowed * (2 * rand(rng) - 1)
    return μ0, r
end

# Draw (μ0, r, direction) such that μ(t) = μ0 + r*t crosses direction * μ_c exactly at a
# uniformly-drawn crossing time t_cross in [margin_frac*T, (1-margin_frac)*T] (leaving room in
# the simulation window to see both the pre-tip drift and the post-tip relaxation).
function sample_tipping_mu(rng::AbstractRNG, T::Real, a::Real; μ_bound::Real = 0.3, margin_frac::Real = 0.2)
    μ_c = μ_critical(a)
    direction = rand(rng, (-1, 1))
    μ0 = μ_bound * (2 * rand(rng) - 1)
    t_cross = T * (margin_frac + (1 - 2 * margin_frac) * rand(rng))
    r = (direction * μ_c - μ0) / t_cross
    return μ0, r, direction, t_cross
end

# Initial condition with c placed on the requested stable branch (±√a, exact only at μ=0 but
# close enough that the fast relaxation of x/y/(q,p) during spin-up dominates any small error).
function initial_condition(params::L96MultiscaleParams, branch::Symbol, rng::AbstractRNG)
    u0 = default_initial_condition(params; rng = rng)
    u0[c_index(params)] = branch == :positive ? sqrt(params.a) : -sqrt(params.a)
    return u0
end

# Run one realization: spin-up (fixed initial μ) then a saved main window. Only x_k(t) is
# treated as observed data (matching Y_{0:T} = {x(t;θ†,u(t))} in the write-up) and is what the
# H_(.) summary statistics are computed on; q, p, c (and the large unresolved fast block y,
# which is never saved at all) are latent and are returned only under "X_latent" for
# ground-truth/diagnostic use (e.g. checking whether a trajectory actually tipped) — a learning
# or calibration procedure should never read "X_latent".
function run_realization(
    base_params::L96MultiscaleParams,
    μ0::Real,
    r::Real,
    branch::Symbol,
    rng::AbstractRNG;
    T_spinup::Real = 30.0,
    T_main::Real = 2000.0,
    dt_save::Real = 0.05,
)
    params = L96MultiscaleParams(;
        K = base_params.K,
        J = base_params.J,
        F0 = base_params.F0,
        χ = base_params.χ,
        hx = base_params.hx,
        hy = base_params.hy,
        ε = base_params.ε,
        ω = base_params.ω,
        γ = base_params.γ,
        α = base_params.α,
        βq = base_params.βq,
        βp = base_params.βp,
        δ = base_params.δ,
        a = base_params.a,
        λ = base_params.λ,
        ρ = base_params.ρ,
        μ = linear_mu(μ0, r),
    )

    u_init = initial_condition(params, branch, rng)
    spinup_sol = solve_l96_multiscale(params, u_init, (0.0, T_spinup))
    u0 = spinup_sol.u[end]

    sol = solve_l96_multiscale(params, u0, (0.0, T_main); saveat = dt_save)

    t_vec = sol.t
    Xfull = reduce(hcat, sol.u)
    X_obs = Xfull[collect(x_indices(params)), :] # rows: x_1..x_K  (observed; used for learning)
    X_latent = Xfull[[q_index(params), p_index(params), c_index(params)], :] # rows: q, p, c (unobserved)
    μ_vec = params.μ.(t_vec)

    stats = Dict(
        "H_mean" => H_mean(X_obs),
        "H_var" => H_var(X_obs),
        "H_quantile" => H_quantile(X_obs, (0.05, 0.95)),
        "H_ac" => H_ac(X_obs),
        "H_spec_fast" => H_spec(X_obs, t_vec, (5.0, 15.0)),      # around the fast-variable scale ~1/ε
        "H_spec_oscillator" => H_spec(X_obs, t_vec, (0.4, 0.8)), # around the (q,p) frequency ω
    )

    return Dict(
        "t" => t_vec,
        "mu" => μ_vec,
        "X_obs" => X_obs,
        "X_latent" => X_latent,
        "K" => params.K,
        "stats" => stats,
        "mu0" => μ0,
        "r" => r,
        "branch" => String(branch),
        "retcode" => String(Symbol(sol.retcode)),
    )
end

# Whether c(t) ever left the stable branch it started on. Because the barrier between the
# stable and unstable branches shrinks continuously to zero as μ approaches ±μ_c, the
# chaotic/noise-like λx̄+ρȳ forcing on c can trigger a real tipping event even when μ(t) never
# deterministically crosses the threshold — so "μ stayed within bound" alone does not guarantee
# a non-tipping realization, and this check is needed on top of it. This is a ground-truth check
# (reads the latent c), used only when constructing/validating the dataset, never by a learner.
function tipped(realization::Dict{String, Any})
    c = realization["X_latent"][end, :]
    return any(sign.(c) .!= sign(c[1]))
end
