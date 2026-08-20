# Generates a single BASELINE realization: μ(t) ≡ μ0 (no exploratory climate forcing, r = 0),
# integrated with an adaptive embedded RK4(5) (Dormand-Prince / "ode45") over a short window at
# fine time resolution, so the fast/chaotic/oscillatory statistics (H_ac, H_spec) are well
# resolved. Saved in the same ensemble ("realizations" vector) format as
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
T_spinup, T_main, dt_save = 30.0, 60.0, 0.02
μ0, r = 0.0, 0.0

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

@info "Baseline run: μ(t) ≡ $(μ0), integrating $(T_main) MTU (after $(T_spinup) MTU spin-up), saved every $(dt_save) MTU"
realization = run_realization(base_params, μ0, r, :positive, MersenneTwister(rng_seed); T_spinup = T_spinup, T_main = T_main, dt_save = dt_save)
realization["seed"] = rng_seed
@info "retcode = $(realization["retcode"]), n_saved_points = $(length(realization["t"])), tipped = $(tipped(realization))"

data_filename = joinpath(output_dir, "l96_multiscale_baseline_$(today()).jld2")
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
@info "saved baseline realization to $(data_filename)"
