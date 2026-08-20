# Diagnostic plots for ONE realization from an ensemble-format output file (as produced by
# `generate_baseline_data.jl`, `generate_tipping_data.jl`, `generate_training_data.jl`, or
# `generate_testing_data.jl`): state-space views (x, oscillator (q,p), climate c) plus the
# H_(.) summary statistics, used to check that the model reproduces the intended timescale
# separation from Table 2 (fast y ~ 0.1 MTU, chaotic x ~ 1 MTU, oscillator (q,p) ~ 10.47 MTU,
# slow climate c ~ 200 MTU).
using JLD2
using Dates

include("PlottingUtils.jl")

output_dir = joinpath(@__DIR__, "output")

# Which source file and which realization within it (1-based index into "realizations") to
# plot. Edit these to look at e.g. the testing ensemble instead: change source_filename to
# "l96_multiscale_testing_$(today()).jld2" and ensemble_index to 1:6.
source_filename = joinpath(output_dir, "l96_multiscale_baseline_$(today()).jld2")
ensemble_index = 1

@info "reading $(source_filename), realization $(ensemble_index)"
data = JLD2.load(source_filename)
realization = data["realizations"][ensemble_index]

tag = "$(splitext(basename(source_filename))[1])_$(ensemble_index)"
plot_realization_diagnostics(realization, output_dir, tag)

@info "All diagnostic plots for $(tag) written to $(output_dir)"
