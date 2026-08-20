# Diagnostic plots for the full training/testing ensembles: an aggregate overview (confirms
# every training realization's μ(t) stays within the bistability threshold ±μ_c so none tip,
# and every testing realization's μ(t) crosses it so all do), plus the full per-realization
# state/statistics diagnostics (via `plot_realization_diagnostics` from `PlottingUtils.jl`) for
# EVERY realization in both ensembles.
# c(t) here is the ground-truth latent climate variable ("X_latent"), plotted only to confirm
# the ensembles behave as intended — a learner only ever sees "X_obs" (x_k(t)) and μ(t).
using Plots
using JLD2
using Dates

include("PlottingUtils.jl")

output_dir = joinpath(@__DIR__, "output")
train_filename = joinpath(output_dir, "l96_multiscale_training_$(today()).jld2")
test_filename = joinpath(output_dir, "l96_multiscale_testing_$(today()).jld2")
@info "reading $(train_filename)"
@info "reading $(test_filename)"

d_train = JLD2.load(train_filename)
d_test = JLD2.load(test_filename)
train_realizations = d_train["realizations"]
test_realizations = d_test["realizations"]

a = 1.0 # matches base_params.a in the generation scripts
μ_c = μ_critical(a)

########################################################################
############################ Aggregate overview ###########################
########################################################################
p1 = plot(xlabel = "t (MTU)", ylabel = "μ(t)", title = "Training: μ(t) (never crosses ±μ_c)", legend = false)
for r in train_realizations
    plot!(p1, r["t"], r["mu"], lw = 1, alpha = 0.7)
end
hline!(p1, [μ_c, -μ_c], linestyle = :dash, color = :red)

p2 = plot(xlabel = "t (MTU)", ylabel = "c(t)", title = "Training: c(t) (stays on its initial branch)", legend = false)
for r in train_realizations
    plot!(p2, r["t"], r["X_latent"][end, :], lw = 1, alpha = 0.7)
end
hline!(p2, [sqrt(a), -sqrt(a)], linestyle = :dash, color = :black)

p3 = plot(xlabel = "t (MTU)", ylabel = "μ(t)", title = "Testing: μ(t) (guaranteed to cross ±μ_c)", legend = false)
for r in test_realizations
    plot!(p3, r["t"], r["mu"], lw = 1, alpha = 0.7)
end
hline!(p3, [μ_c, -μ_c], linestyle = :dash, color = :red)

p4 = plot(xlabel = "t (MTU)", ylabel = "c(t)", title = "Testing: c(t) (forced critical transitions)", legend = false)
for r in test_realizations
    plot!(p4, r["t"], r["X_latent"][end, :], lw = 1, alpha = 0.7)
end
hline!(p4, [sqrt(a), -sqrt(a)], linestyle = :dash, color = :black)

ensemble_plot = plot(p1, p2, p3, p4, layout = (4, 1), size = (900, 1400), left_margin = 8Plots.mm)
savefig(ensemble_plot, joinpath(output_dir, "diagnostic_ensembles.png"))
@info "saved diagnostic_ensembles.png"

########################################################################
################### Per-realization diagnostics, every run ###############
########################################################################
for (i, r) in enumerate(train_realizations)
    plot_realization_diagnostics(r, output_dir, "training_$(i)")
end
for (i, r) in enumerate(test_realizations)
    plot_realization_diagnostics(r, output_dir, "testing_$(i)")
end

@info "All per-realization diagnostics for $(length(train_realizations)) training and $(length(test_realizations)) testing runs written to $(output_dir)"
