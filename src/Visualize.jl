module Visualize

"""
    plot_parameter_distribution(fig, pd; constrained=true, n_sample=1e4, rng=Random.GLOBAL_RNG)

Plot histogram marginals of the parameter distribution `pd` on `fig`.

Each dimension of `pd` is drawn as a separate histogram panel. When
`constrained = true` (default), samples are mapped from the unconstrained
space to the constrained space before plotting.

Requires a Makie backend (e.g. `using CairoMakie`) to be loaded.

# Arguments
- `fig`: a `Makie.Figure`, `Makie.GridLayout`, `Makie.GridPosition`, or
  `Makie.GridSubposition` that receives the plots.
- `pd`: the `ParameterDistribution` (or `ParameterDistributionType`) to plot.
- `constrained`: if `true`, transform samples to constrained space before
  plotting.
- `n_sample`: number of Monte Carlo samples used to build the histograms.
- `rng`: random-number generator.
"""
function plot_parameter_distribution end

"""
    plot_error_over_iters(gridposition, ekp; error_metric=nothing, kwargs...)

Plot the error metric of `ekp` against the iteration index on `gridposition`.

The x-axis shows iteration numbers; the y-axis shows the chosen error metric.
`error_metric` selects which metric to display; when `nothing` the metric is
chosen automatically based on the process type. Additional keyword arguments
are forwarded to `Makie.Lines`.

Requires a Makie backend to be loaded.
"""
function plot_error_over_iters end

"""
    plot_error_over_iters!(axis, ekp; error_metric=nothing, kwargs...)

Plot the error metric of `ekp` against the iteration index into an existing `axis`.

Mutating variant of [`plot_error_over_iters`](@ref). The `axis` argument must
be a `Makie.AbstractAxis`. Additional keyword arguments are forwarded to
`Makie.Lines`.

Requires a Makie backend to be loaded.
"""
function plot_error_over_iters! end

"""
    plot_error_over_time(gridposition, ekp; error_metric=nothing, kwargs...)

Plot the error metric of `ekp` against algorithm time on `gridposition`.

The x-axis shows the accumulated pseudo-time (sum of step sizes `Δt`); the
y-axis shows the chosen error metric. `error_metric` selects which metric to
display; when `nothing` the metric is chosen automatically based on the
process type. Additional keyword arguments are forwarded to `Makie.Lines`.

Requires a Makie backend to be loaded.
"""
function plot_error_over_time end

"""
    plot_error_over_time!(axis, ekp; error_metric=nothing, kwargs...)

Plot the error metric of `ekp` against algorithm time into an existing `axis`.

Mutating variant of [`plot_error_over_time`](@ref). The `axis` argument must
be a `Makie.AbstractAxis`. Additional keyword arguments are forwarded to
`Makie.Lines`.

Requires a Makie backend to be loaded.
"""
function plot_error_over_time! end

"""
    plot_ϕ_over_iters(gridposition, ekp, prior, dim_idx; kwargs...)
    plot_ϕ_over_iters(gridpositions, ekp, prior, name; kwargs...)

Plot ensemble members of constrained parameter `dim_idx` (or all dimensions of
distribution `name`) against the iteration index.

Each ensemble member is shown as a scatter point. The first form plots a single
parameter dimension on `gridposition`; the second form iterates over
`gridpositions` and plots every dimension of the named sub-distribution.
Additional keyword arguments are forwarded to `Makie.Scatter`.

Requires a Makie backend to be loaded.

# Arguments
- `gridposition`: a `Makie.GridPosition` (or compatible) receiving the plot.
- `gridpositions`: an iterable of grid positions, one per dimension of `name`.
- `ekp`: the `EnsembleKalmanProcess` providing ensemble trajectories.
- `prior`: the `ParameterDistribution` used to map unconstrained to constrained
  space.
- `dim_idx`: integer index into the full parameter vector.
- `name`: name string of a sub-distribution in `prior`.
"""
function plot_ϕ_over_iters end

"""
    plot_ϕ_over_iters!(axis, ekp, prior, dim_idx; kwargs...)

Plot ensemble members of constrained parameter `dim_idx` against the iteration
index into an existing `axis`.

Mutating variant of [`plot_ϕ_over_iters`](@ref). Additional keyword arguments
are forwarded to `Makie.Scatter`.

Requires a Makie backend to be loaded.
"""
function plot_ϕ_over_iters! end

"""
    plot_ϕ_over_time(gridposition, ekp, prior, dim_idx; kwargs...)
    plot_ϕ_over_time(gridpositions, ekp, prior, name; kwargs...)

Plot ensemble members of constrained parameter `dim_idx` (or all dimensions of
distribution `name`) against algorithm time.

Each ensemble member is shown as a scatter point. The first form plots a single
parameter dimension on `gridposition`; the second form iterates over
`gridpositions` and plots every dimension of the named sub-distribution.
Additional keyword arguments are forwarded to `Makie.Scatter`.

Requires a Makie backend to be loaded.

# Arguments
- `gridposition`: a `Makie.GridPosition` (or compatible) receiving the plot.
- `gridpositions`: an iterable of grid positions, one per dimension of `name`.
- `ekp`: the `EnsembleKalmanProcess` providing ensemble trajectories.
- `prior`: the `ParameterDistribution` used to map unconstrained to constrained
  space.
- `dim_idx`: integer index into the full parameter vector.
- `name`: name string of a sub-distribution in `prior`.
"""
function plot_ϕ_over_time end

"""
    plot_ϕ_over_time!(axis, ekp, prior, dim_idx; kwargs...)

Plot ensemble members of constrained parameter `dim_idx` against algorithm
time into an existing `axis`.

Mutating variant of [`plot_ϕ_over_time`](@ref). Additional keyword arguments
are forwarded to `Makie.Scatter`.

Requires a Makie backend to be loaded.
"""
function plot_ϕ_over_time! end

"""
    plot_ϕ_mean_over_iters!(axis, ekp, prior, dim_idx; plot_std=false, kwargs...)

Plot the ensemble mean of constrained parameter `dim_idx` against the iteration
index into an existing `axis`.

When `plot_std = true`, a band spanning one standard deviation around the mean
is also drawn. Additional keyword arguments are forwarded to `Makie.Lines` (and
to `Makie.Band` when `plot_std = true`).

Requires a Makie backend to be loaded.
"""
function plot_ϕ_mean_over_iters! end

"""
    plot_ϕ_mean_over_iters(gridposition, ekp, prior, dim_idx; plot_std=false, kwargs...)
    plot_ϕ_mean_over_iters(gridpositions, ekp, prior, name; kwargs...)

Plot the ensemble mean of constrained parameter `dim_idx` (or all dimensions of
distribution `name`) against the iteration index.

The first form creates a new axis inside `gridposition`; the second form
iterates over `gridpositions` and plots every dimension of the named
sub-distribution. When `plot_std = true`, a band spanning one standard
deviation around the mean is also drawn. Additional keyword arguments are
forwarded to `Makie.Lines` (and to `Makie.Band` when `plot_std = true`).

Requires a Makie backend to be loaded.
"""
function plot_ϕ_mean_over_iters end

"""
    plot_ϕ_mean_over_time!(axis, ekp, prior, dim_idx; plot_std=false, kwargs...)

Plot the ensemble mean of constrained parameter `dim_idx` against algorithm
time into an existing `axis`.

When `plot_std = true`, a band spanning one standard deviation around the mean
is also drawn. Additional keyword arguments are forwarded to `Makie.Lines` (and
to `Makie.Band` when `plot_std = true`).

Requires a Makie backend to be loaded.
"""
function plot_ϕ_mean_over_time! end

"""
    plot_ϕ_mean_over_time(gridposition, ekp, prior, dim_idx; plot_std=false, kwargs...)
    plot_ϕ_mean_over_time(gridpositions, ekp, prior, name; kwargs...)

Plot the ensemble mean of constrained parameter `dim_idx` (or all dimensions of
distribution `name`) against algorithm time.

The first form creates a new axis inside `gridposition`; the second form
iterates over `gridpositions` and plots every dimension of the named
sub-distribution. When `plot_std = true`, a band spanning one standard
deviation around the mean is also drawn. Additional keyword arguments are
forwarded to `Makie.Lines` (and to `Makie.Band` when `plot_std = true`).

Requires a Makie backend to be loaded.
"""
function plot_ϕ_mean_over_time end

end
