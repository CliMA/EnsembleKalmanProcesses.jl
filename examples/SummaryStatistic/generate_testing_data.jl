# Generates a TESTING ensemble: realizations of the observable trajectory x(t) (plus
# oscillator/climate variables q,p,c) driven by exploratory forcing trajectories
# μ(t) = μ0 + r*t, each constructed to be GUARANTEED to cross the bistability threshold ±μ_c
# within the simulation window, i.e. every realization undergoes a forced tipping event. This
# is meant to test whether statistics/calibration learned only from the non-crossing training
# ensemble (see `generate_training_data.jl`) generalize to (or anticipate) behaviour at/beyond
# the bifurcation.
using Random
using JLD2
using Dates

include("EnsembleRunner.jl")

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

rng_seed_init = 900
N_test = 6
T_spinup, T_main, dt_save = 30.0, 2000.0, 0.05

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

sampling_rng = MersenneTwister(rng_seed_init)
realizations = Vector{Dict{String, Any}}(undef, N_test)
for i in 1:N_test
    member_seed = rng_seed_init + i
    μ0, r, direction, t_cross = sample_tipping_mu(sampling_rng, T_main, base_params.a)
    # direction=+1 crosses the upper threshold, which annihilates the negative branch -> start there.
    # direction=-1 crosses the lower threshold, which annihilates the positive branch -> start there.
    branch = direction > 0 ? :negative : :positive
    @info "Testing realization $i/$(N_test): μ0=$(round(μ0, digits = 3)), r=$(round(r, sigdigits = 3)), branch=$branch, t_cross≈$(round(t_cross, digits = 1))"
    realizations[i] = run_realization(
        base_params,
        μ0,
        r,
        branch,
        MersenneTwister(member_seed);
        T_spinup = T_spinup,
        T_main = T_main,
        dt_save = dt_save,
    )
    realizations[i]["seed"] = member_seed
    realizations[i]["t_cross"] = t_cross
end

data_filename = joinpath(output_dir, "l96_multiscale_testing_$(today()).jld2")
JLD2.save(
    data_filename,
    "realizations",
    realizations,
    "N_test",
    N_test,
    "T_spinup",
    T_spinup,
    "T_main",
    T_main,
    "dt_save",
    dt_save,
    "rng_seed_init",
    rng_seed_init,
)
@info "saved $(N_test) tipping testing realizations to $(data_filename)"
