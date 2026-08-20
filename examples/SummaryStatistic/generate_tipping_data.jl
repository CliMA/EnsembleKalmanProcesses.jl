# Generates a single TIPPING-DEMONSTRATION realization: μ(t) = μ0 + r*t with fixed, specific
# (μ0, r) chosen to cross the saddle-node threshold +μ_c = 2/(3√3) ≈ 0.385 (for a=1) partway
# through the window, starting on the branch that gets annihilated by the crossing. Integrated
# over a long window at coarser time resolution (appropriate for the slow, δ-scale climate
# dynamics). Saved in the same ensemble ("realizations" vector) format as
# `generate_training_data.jl` / `generate_testing_data.jl` so the same diagnostic scripts
# (`diagnostics_plots.jl`, `diagnostics_ensemble_plots.jl`) work on it unchanged.
using Random
using JLD2
using Dates

include("EnsembleRunner.jl")

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

rng_seed = 42
T_spinup, T_main, dt_save = 30.0, 2000.0, 0.5
μ0, r = 0.3, 1e-4 # r within the O(1e-4 - 1e-3) range from Table 1

base_params = L96MultiscaleParams(
    K = 8,
    J = 32,
    F0 = 18.0,
    χ = 0.1,
    hx = 1.0,
    hy = 1.0,
    ε = 0.1,
    ω = 0.6,
    γ = 0.02,
    α = 0.1,
    βq = 0.05,
    βp = 0.05,
    δ = 0.005,
    a = 1.0,
    λ = 0.05,
    ρ = 0.05,
)

# μ0+r*t increases, so it crosses the upper threshold +μ_c, which annihilates the negative
# branch -> start there.
@info "Tipping demo: μ(t) = $(μ0) + $(r)*t, integrating $(T_main) MTU (after $(T_spinup) MTU spin-up), saved every $(dt_save) MTU"
realization = run_realization(base_params, μ0, r, :negative, MersenneTwister(rng_seed); T_spinup = T_spinup, T_main = T_main, dt_save = dt_save)
realization["seed"] = rng_seed
@info "retcode = $(realization["retcode"]), n_saved_points = $(length(realization["t"])), tipped = $(tipped(realization))"

data_filename = joinpath(output_dir, "l96_multiscale_tipping_$(today()).jld2")
JLD2.save(
    data_filename,
    "realizations",
    [realization],
    "N",
    1,
    "T_spinup",
    T_spinup,
    "T_main",
    T_main,
    "dt_save",
    dt_save,
    "rng_seed",
    rng_seed,
)
@info "saved tipping-demonstration realization to $(data_filename)"
