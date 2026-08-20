# Shared per-realization diagnostic plotting, used by both `diagnostics_plots.jl` (a single
# chosen realization) and `diagnostics_ensemble_plots.jl` (looped over every realization in an
# ensemble file). Operates on the realization Dict produced by `run_realization` in
# `EnsembleRunner.jl`: "X_obs" (observed x_k(t)) drives the H_(.) summary-statistic panels;
# "X_latent" (q, p, c) and "mu" are shown only for physical validation (ground truth), never as
# something a learner would see.
using Plots
using Statistics

include("EnsembleRunner.jl")

function plot_realization_diagnostics(
    realization::Dict,
    output_dir::String,
    tag::String;
    a::Real = 1.0,
    ω::Real = 0.6,
    ε::Real = 0.1,
)
    t = realization["t"]
    X = realization["X_obs"]     # K x n_time
    Xl = realization["X_latent"] # 3 x n_time: q, p, c
    mu = realization["mu"]
    K = realization["K"]
    stats = realization["stats"]
    dt = t[2] - t[1]
    μ_c = μ_critical(a)

    ########################################################################
    ############################ State diagnostics ###########################
    ########################################################################

    p1 = heatmap(t, 1:K, X, xlabel = "t (MTU)", ylabel = "k", title = "x_k(t)", color = :balance)

    p2 = plot(t, X[1, :], label = "x_1", xlabel = "t (MTU)", title = "Resolved slow variables", lw = 1)
    if K >= 2
        plot!(p2, t, X[2, :], label = "x_2", lw = 1)
    end

    p3 = plot(t, mu, xlabel = "t (MTU)", ylabel = "μ(t)", title = "Exploratory forcing μ(t)", legend = false, lw = 1.5)
    hline!(p3, [μ_c, -μ_c], linestyle = :dash, color = :red)

    p4 = plot(t, Xl[1, :], label = "q", xlabel = "t (MTU)", title = "Oscillator (q,p) [latent]", lw = 1.5)
    plot!(p4, t, Xl[2, :], label = "p", lw = 1.5)
    p5 = plot(Xl[1, :], Xl[2, :], xlabel = "q", ylabel = "p", title = "(q,p) phase portrait [latent]", legend = false, lw = 0.5)

    p6 = plot(t, Xl[3, :], xlabel = "t (MTU)", ylabel = "c", title = "Climate variable c(t) [latent]", legend = false, lw = 1.5)
    hline!(p6, [sqrt(a), -sqrt(a)], linestyle = :dash, color = :black)

    state_plot = plot(p1, p2, p3, p4, p5, p6, layout = (6, 1), size = (900, 1900), left_margin = 8Plots.mm)
    savefig(state_plot, joinpath(output_dir, "diagnostic_state_$(tag).png"))
    @info "saved diagnostic_state_$(tag).png"

    ########################################################################
    ###################### Summary-statistic diagnostics ######################
    ########################################################################

    p7 = scatter(1:K, stats["H_mean"], xlabel = "k", ylabel = "value", title = "H_mean(x_k)", legend = false)
    p8 = scatter(1:K, stats["H_var"], xlabel = "k", ylabel = "value", title = "H_var(x_k)", legend = false)

    Q = stats["H_quantile"]
    p9 = plot(t, X[1, :], label = "x_1(t)", xlabel = "t (MTU)", title = "x_1 with H_quantile(0.05,0.95) band", lw = 1)
    hline!(p9, [Q[1, 1], Q[1, 2]], linestyle = :dash, color = :black, label = "Q(0.05), Q(0.95)")

    # Autocorrelation of x_1 (observed) vs q (latent, shown only to validate the oscillator
    # timescale), with H_ac(x_1) (lag-1 value) marked.
    maxlag = min(round(Int, 25 / dt), size(X, 2) - 1)
    acf = autocorrelation_function(vcat(X[1:1, :], Xl[1:1, :]), maxlag)
    lags = (0:maxlag) .* dt
    p10 = plot(lags, acf[1, :], label = "acf(x_1)", xlabel = "lag (MTU)", title = "Autocorrelation function")
    plot!(p10, lags, acf[2, :], label = "acf(q) [latent]")
    scatter!(p10, [lags[2]], [stats["H_ac"][1]], label = "H_ac(x_1)", color = :blue)

    # Power spectral density of x_1(t) vs q(t) [latent], marking the expected fast (~1/ε) and
    # oscillator (ω) angular frequencies.
    freqs, psd = power_spectrum(vcat(X[1:1, :], Xl[1:1, :]), t)
    pos = freqs .>= 0
    xmax = min(20.0, maximum(freqs[pos]))
    p11 = plot(
        freqs[pos],
        psd[1, pos],
        yscale = :log10,
        xlabel = "ω (rad/MTU)",
        ylabel = "PSD",
        label = "x_1(t)",
        title = "Power spectral density",
        xlims = (0, xmax),
        ylims = (1e-6, 10),
    )
    plot!(p11, freqs[pos], psd[2, pos], label = "q(t) [latent]")
    vline!(p11, [ω], linestyle = :dash, color = :black, label = "ω = $ω")
    vline!(p11, [1 / ε], linestyle = :dash, color = :red, label = "1/ε = $(1/ε)")

    stats_plot = plot(p7, p8, p9, p10, p11, layout = (5, 1), size = (900, 1800), left_margin = 8Plots.mm)
    savefig(stats_plot, joinpath(output_dir, "diagnostic_statistics_$(tag).png"))
    @info "saved diagnostic_statistics_$(tag).png"

    return nothing
end
