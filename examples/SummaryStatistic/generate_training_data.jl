# Generates a TRAINING ensemble: many realizations of the observable trajectory x(t) (plus
# oscillator/climate variables q,p,c) driven by different exploratory forcing trajectories
# μ(t) = μ0 + r*t, each constructed so that μ(t) never crosses the bistability threshold ±μ_c.
# No tipping occurs in any training realization — per the write-up's remark that observing
# different u(t) away from a bifurcation can still carry information about it, this is meant
# to be the "normal regime" data a calibration/training procedure has access to.
using Random
using JLD2
using Dates

include("EnsembleRunner.jl")

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

rng_seed_init = 100
N_train = 15
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
realizations = Vector{Dict{String, Any}}(undef, N_train)
max_attempts = 6
for i in 1:N_train
    member_seed = rng_seed_init + i
    local_result = nothing
    for attempt in 1:max_attempts
        # Shrink the sampling bound on each retry so that a realization which tipped (via
        # chaotic-noise-triggered early escape near the shrinking bistable barrier, not
        # necessarily a deterministic μ-threshold crossing) is very likely to succeed on the
        # next, more conservative, attempt; attempt == max_attempts falls back to μ ≡ 0
        # (guaranteed far from either threshold) so the training set is always tipping-free.
        μ0, r = attempt < max_attempts ? sample_nontipping_mu(sampling_rng, T_main, base_params.a; safety = 0.6 / attempt) : (0.0, 0.0)
        branch = rand(sampling_rng, (:positive, :negative))
        candidate = run_realization(
            base_params,
            μ0,
            r,
            branch,
            MersenneTwister(member_seed);
            T_spinup = T_spinup,
            T_main = T_main,
            dt_save = dt_save,
        )
        if !tipped(candidate)
            @info "Training realization $i/$(N_train) (attempt $attempt): μ0=$(round(μ0, digits = 3)), r=$(round(r, sigdigits = 3)), branch=$branch"
            local_result = candidate
            break
        else
            @warn "Training realization $i attempt $attempt tipped despite μ staying within the deterministic bound (μ0=$(round(μ0, digits = 3)), r=$(round(r, sigdigits = 3)), branch=$branch); resampling with a smaller bound"
        end
    end
    local_result["seed"] = member_seed
    realizations[i] = local_result
end

data_filename = joinpath(output_dir, "l96_multiscale_training_$(today()).jld2")
JLD2.save(
    data_filename,
    "realizations",
    realizations,
    "N_train",
    N_train,
    "T_spinup",
    T_spinup,
    "T_main",
    T_main,
    "dt_save",
    dt_save,
    "rng_seed_init",
    rng_seed_init,
)
@info "saved $(N_train) non-tipping training realizations to $(data_filename)"
