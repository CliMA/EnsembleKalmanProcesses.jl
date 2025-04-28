# Visualization

We provide some simple plotting features to help users diagnose convergence in
the algorithm. Through

- Visualizing priors
- Visualizing slices of parameter ensembles over iterations (or algorithm time) over input space
- Visualizing the loss function or other computed error metrics from [`compute_error!`](@ref)

*The following documentation provides the overview of what is currently implemented for Plots or Makie backends, these will be expanded in due course.*

## Plots.jl

!!! note "Add Plots.jl to your Project.toml"
    To enable plotting by Plots.jl, use `using Plots`.

Plotting using Plots.jl supports only plotting distributions. See the example
below.

```@example
using EnsembleKalmanProcesses.ParameterDistributions
prior_u1 = constrained_gaussian("positive_with_mean_2", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("four_with_spread_5", 0, 5, -Inf, Inf, repeats=4)
prior = combine_distributions([prior_u1, prior_u2])

using Plots
p = plot(prior)
```

## Makie.jl

!!! note "Add a Makie-backend package to your Project.toml"
    Import one of the Makie backends (GLMakie, CairoMakie, WGLMakie, RPRMakie,
    etc.) to enable these functions!

Plotting functionality is provided by Makie.jl through a package extension. See
the [documentation](@ref Visualize) for a list of all the available plotting
functions.

```@setup makie_plots
# Fix random seed, so plots don't change
import Random
rng_seed = 1234
rng = Random.MersenneTwister(rng_seed)

using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

G(u) =
    [1.0 / abs(u[1]), sum(u[2:3]), u[3], u[1]^2 - u[2] - u[3], u[1], 5.0] .+ 0.1 * randn(6)
true_u = [3, 1, 2]
y = G(true_u)
Γ = (0.1)^2 * I

prior_u1 = constrained_gaussian("positive_with_mean_1", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("two_with_spread_2", 0, 5, -Inf, Inf, repeats = 2)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 30
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
ekp = EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(), verbose = true)

N_iterations = 10
for i = 1:N_iterations
    params_i = get_ϕ_final(prior, ekp)

    G_matrix = hcat(
        [G(params_i[:, i]) for i = 1:N_ensemble]..., # Parallelize here!
    )

    update_ensemble!(ekp, G_matrix)
end
```

### Plot priors

The function `plot_parameter_distribution` will plot the marginal histograms for
all dimensions of the parameter distribution.

```@example makie_plots
using CairoMakie # load a Makie backend
import EnsembleKalmanProcesses.Visualize as viz

fig_priors = CairoMakie.Figure(size = (500, 400))
viz.plot_parameter_distribution(fig_priors[1, 1], prior)

fig_priors
```

### Plot errors

The functions `plot_error_over_iters` and `plot_error_over_time` and the
mutating versions can be used to plot the errors over the number of iterations
or time respectively. All keyword arguments supported by
[`Makie.lines`](https://docs.makie.org/dev/reference/plots/lines) are supported
by these functions too.

Any of the stored error metrics, computed by EnsembleKalmanProcess can be
plotted by the `error_metric="metric-name"` keyword argument, . See
[`compute_error!`](@ref) for a list of the computed error metrics and their
names.

```@example makie_plots
# Assume that ekp is the EnsembleKalmanProcess object
using CairoMakie # load a Makie backend
import EnsembleKalmanProcesses.Visualize as viz

fig_errors = CairoMakie.Figure(size = (300 * 2, 300 * 1))
viz.plot_error_over_iters(fig_errors[1, 1], ekp, color = :tomato)
# Error plotting functions support plotting different errors through the
# error_metric keyword argument
ax1 = CairoMakie.Axis(fig_errors[1, 2], title = "Average RMSE over iterations", yscale = Makie.pseudolog10)
viz.plot_error_over_time!(
    ax1,
    ekp,
    linestyle = :dashdotdot,
    error_metric = "avg_rmse",
)
fig_errors
```

### Plot constrained parameters

#### Plot by a slice of the parameter

The functions `plot_ϕ_over_iters` and `plot_ϕ_over_time` and the mutating
versions can be used to plot a scatter plot of an individual parameter (dimension) over
the number of iterations or time respectively. All keyword arguments supported
by [`Makie.Scatter`](https://docs.makie.org/dev/reference/plots/scatter) are
supported by these functions too.

Furthermore, there are also `plot_ϕ_mean_over_iters` and `plot_ϕ_mean_over_time`
and the mutating versions can be used to plot the mean parameter (dimension)
over the number of iterations or time respectively. All keyword arguments
supported by [`Makie.Scatter`](https://docs.makie.org/dev/reference/plots/lines)
are supported by these functions too. Furthermore, there is the keyword argument
`plot_std = true` which can be used to also plot the standard deviation as well.
To support plotting both mean and standard deviation, there are also the keyword
arguments `line_kwargs` and `band_kwargs` for adjusting how the mean and
standard deviation are plotted. Any keyword arguments accepted by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines) can be
passed to `line_kwargs` and any keyword arguments accepted by
[`Makie.Band`](https://docs.makie.org/dev/reference/plots/band) can be passed to
`band_kwargs`.

```@example makie_plots
using CairoMakie # load a Makie backend
import EnsembleKalmanProcesses.Visualize as viz

fig_ϕ = CairoMakie.Figure(size = (300 * 2, 300 * 2))
EnsembleKalmanProcesses.Visualize.plot_ϕ_over_time(fig_ϕ[1, 1], ekp, prior, 1, axis = (xscale = Makie.pseudolog10,))
ax1 = CairoMakie.Axis(fig_ϕ[1, 2])
viz.plot_ϕ_mean_over_iters!(ax1, ekp, prior, 2)
viz.plot_ϕ_mean_over_iters(
            fig_ϕ[2, 1],
            ekp,
            prior,
            3,
            plot_std = true,
            line_kwargs = (linestyle = :dash,),
            band_kwargs = (alpha = 0.2,),
            color = :purple,
        )
fig_ϕ
```
