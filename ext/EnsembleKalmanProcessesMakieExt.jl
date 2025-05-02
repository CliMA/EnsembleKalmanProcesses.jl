module EnsembleKalmanProcessesMakieExt

using Makie

import EnsembleKalmanProcesses.Visualize as Visualize
import EnsembleKalmanProcesses:
    EnsembleKalmanProcess,
    get_N_iterations,
    get_Δt,
    get_algorithm_time,
    get_error_metrics,
    get_ϕ,
    get_ϕ_mean,
    get_process,
    get_prior_mean,
    get_prior_cov
using EnsembleKalmanProcesses.ParameterDistributions
import Statistics: std

using Random

# Plotting recipe does not support plotting multiple plots, so this is a convience wrapper that
# plot the priors with no support for passing in keyword arguments
"""
    Visualize.plot_parameter_distribution(fig::Union{Makie.Figure, Makie.GridLayout, Makie.GridPosition, Makie.GridSubposition},
                           pd::ParameterDistribution;
                           constrained = true,
                           n_sample = 1e4,
                           rng = Random.GLOBAL_RNG)

Plot the distributions `pd` on `fig`.
"""
function Visualize.plot_parameter_distribution(
    fig::Union{Makie.Figure, Makie.GridLayout, Makie.GridPosition, Makie.GridSubposition},
    pd::ParameterDistribution;
    constrained = true,
    n_sample = 1e4,
    rng = Random.GLOBAL_RNG,
)
    samples = sample(rng, pd, Int(n_sample))
    if constrained
        samples = transform_unconstrained_to_constrained(pd, samples)
    end
    n_plots = ndims(pd)
    batches = batch(pd)

    # Determine the number of cols for the figure
    cols = Int(ceil(sqrt(n_plots)))

    for i in 1:n_plots
        r = div(i - 1, cols) + 1
        c = (i - 1) % cols + 1
        batch_id = [j for j = 1:length(batches) if i ∈ batches[j]][1]
        dim_in_batch = i - minimum(batches[batch_id]) + 1 # i.e. if i=5 in batch 3:6, this would be "3"
        ax = Makie.Axis(
            fig[r, c],
            title = pd.name[batch_id] * " (dim " * string(dim_in_batch) * ")",
            xgridvisible = false,
            ygridvisible = false,
        )
        Makie.ylims!(ax, 0.0, nothing)
        Makie.hist!(
            ax,
            samples[i, :],
            normalization = :pdf,
            color = Makie.Cycled(batch_id),
            strokecolor = (:black, 0.7),
            strokewidth = 0.7,
            bins = Int(floor(sqrt(n_sample))),
        )
    end
    return nothing
end

"""
    Visualize.plot_parameter_distribution(fig::Union{Makie.Figure, Makie.GridLayout},
                           pd::PDT;
                           constrained = true,
                           n_sample = 1e4,
                           rng = Random.GLOBAL_RNG)
                           where {PDT <: ParameterDistributionType}

Plot the distribution on `fig`.
"""
function Visualize.plot_parameter_distribution(
    fig::Union{Makie.Figure, Makie.GridLayout, Makie.GridPosition, Makie.GridSubposition},
    d::PDT;
    n_sample = 1e4,
    rng = Random.GLOBAL_RNG,
) where {PDT <: ParameterDistributionType}
    samples = sample(rng, pd, Int(n_sample))

    n_plots = ndims(d)

    # Determine the number of cols for the figure
    cols = Int(ceil(sqrt(n_plots)))

    for i in 1:n_plots
        r = div(i - 1, cols) + 1
        c = (i - 1) % cols + 1
        batch_id = [j for j = 1:length(batches) if i ∈ batches[j]][1]
        # i.e. if i=5 in batch 3:6, this would be "3"
        dim_in_batch = i - minimum(batches[batch_id]) + 1
        ax = Makie.Axis(
            fig[r, c],
            title = pd.name[batch_id] * " (dim " * string(dim_in_batch) * ")",
            xgridvisible = false,
            ygridvisible = false,
        )
        Makie.ylims!(ax, 0.0, nothing)
        Makie.hist!(
            ax,
            samples[i, :],
            normalization = :pdf,
            color = Makie.Cycled(batch_id),
            strokecolor = (:black, 0.7),
            strokewidth = 0.7,
            bins = Int(floor(sqrt(n_sample))),
        )
    end
    return nothing
end

# Define plot recipe for iterations or time vs errors line plots
# Define the function erroranditersortime whose functionality is
# implemented by specializing
# Makie.plot!(plot::ConstrainedParamsAndItersOrTime)
@recipe(ErrorAndItersOrTime, ekp) do scene
    Attributes(linewidth = 4.5)
end

"""
    Visualize.plot_error_over_iters(gridposition, ekp; kwargs...)

Plot the errors from `ekp` against the number of iterations on `gridposition`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines).
"""
Visualize.plot_error_over_iters(args...; kwargs...) = _plot_error_over_iters(erroranditersortime, args...; kwargs...)

"""
    Visualize.plot_error_over_iters!(axis, ekp; kwargs...)

Plot the errors from `ekp` against the number of iterations on `axis`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines).
"""
Visualize.plot_error_over_iters!(args...; kwargs...) = _plot_error_over_iters(erroranditersortime!, args...; kwargs...)

"""
    _plot_errors_over_iters(plot_fn, args...; kwargs...)

Helper function to plot the errors against the number of iterations.

The arguments one can expect is `fig_or_ax` and `ekp` or the last argument.
"""
function _plot_error_over_iters(plot_fn, args...; kwargs...)
    ekp = last(args)
    iters = 1:get_N_iterations(ekp)
    error_metric = _find_appropriate_error(ekp, get(kwargs, :error_metric, nothing))
    plot = plot_fn(args...; xvals = iters, error_metric = error_metric, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    _modify_axis!(
        ax;
        xlabel = "Iterations",
        title = "Error over iterations",
        ylabel = "$error_metric",
        xticks = 1:get_N_iterations(ekp),
    )
    return plot
end

"""
    Visualize.plot_error_over_time(gridposition, ekp; kwargs...)

Plot the errors from `ekp` against time on `gridposition`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines).
"""
Visualize.plot_error_over_time(args...; kwargs...) = _plot_error_over_time(erroranditersortime, args...; kwargs...)

"""
    Visualize.plot_error_over_time!(axis, ekp; kwargs...)

Plot the errors from `ekp` against time on `axis`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines).
"""
Visualize.plot_error_over_time!(args...; kwargs...) = _plot_error_over_time(erroranditersortime!, args...; kwargs...)

"""
    _plot_error_over_time(plot_fn, fig_or_ax, ekp; kwargs...)

Helper function to plot errors over time.

The arguments one can expect is `fig_or_ax` and `ekp` or the last argument.
"""
function _plot_error_over_time(plot_fn, args...; kwargs...)
    ekp = last(args)
    times = get_algorithm_time(ekp)
    error_metric = _find_appropriate_error(ekp, get(kwargs, :error_metric, nothing))
    plot = plot_fn(args...; xvals = times, error_metric = error_metric, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    _modify_axis!(ax; xlabel = "Time", title = "Error over time", ylabel = "$error_metric")
    return plot
end

"""
    Makie.plot!(plot::ErrorAndItersOrTime)

Plot the errors against iteration or time.
"""
function Makie.plot!(plot::ErrorAndItersOrTime)
    xvals = plot.xvals[]
    errors = get_error_metrics(plot.ekp[])[plot.error_metric[]]
    valid_attributes = Makie.shared_attributes(plot, Makie.Lines)
    Makie.lines!(plot, xvals, errors; valid_attributes...)
    return plot
end

# Define plot recipe for iteration or time vs constrained parameters scatter plots
# Define the function constrainedparamsanditersortime whose functionality is
# implemented by specializing Makie.plot!(plot::ConstrainedParamsAndItersOrTime)
@recipe(ConstrainedParamsAndItersOrTime, ekp, prior, dim_idx) do scene
    Theme()
end

"""
    Visualize.plot_ϕ_over_iters(gridposition, ekp, prior, dim_idx; kwargs...)

Plot the constrained parameter of index `dim_idx` against time on `gridposition`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
(`Makie.Scatter`)[https://docs.makie.org/dev/reference/plots/scatter].
"""
Visualize.plot_ϕ_over_iters(args...; kwargs...) =
    _plot_ϕ_over_iters(constrainedparamsanditersortime, args...; kwargs...)

"""
    Visualize.plot_ϕ_over_iters!(axis, ekp, prior, dim_idx; kwargs...)

Plot the constrained parameter of index `dim_idx` against time on `axis`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
(`Makie.Scatter`)[https://docs.makie.org/dev/reference/plots/scatter].
"""
Visualize.plot_ϕ_over_iters!(args...; kwargs...) =
    _plot_ϕ_over_iters(constrainedparamsanditersortime!, args...; kwargs...)

"""
    _plot_ϕ_over_iters(plot_fn, args...; kwargs...)

Helper function to plot constrained parameters against the number of iterations.

The arguments one can expect is `fig_or_ax`, `ekp`, `prior`, and `dim_idx` or
the last three arguments.
"""
function _plot_ϕ_over_iters(plot_fn, args...; kwargs...)
    ekp, prior, dim_idx = last(args, 3)
    iters = collect(0:get_N_iterations(ekp))
    plot = plot_fn(args...; xvals = iters, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    _modify_axis!(
        ax;
        xlabel = "Iterations",
        title = "ϕ over iterations [dim $(_get_dim_of_dist(prior, dim_idx))]",
        ylabel = "$(_get_prior_name(prior, dim_idx))",
        xticks = 0:get_N_iterations(ekp),
    )
    return plot
end

"""
    Visualize.plot_ϕ_over_time(gridposition, ekp, prior, dim_idx; kwargs...)

Plot the constrained parameter of index `dim_idx` against time on `gridposition`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
(`Makie.Scatter`)[https://docs.makie.org/dev/reference/plots/scatter].
"""
Visualize.plot_ϕ_over_time(args...; kwargs...) = _plot_ϕ_over_time(constrainedparamsanditersortime, args...; kwargs...)

"""
    Visualize.plot_ϕ_over_time!(axis, ekp, prior, dim_idx; kwargs...)

Plot the constrained parameter of index `dim_idx` against time on `axis`.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
(`Makie.Scatter`)[https://docs.makie.org/dev/reference/plots/scatter].
"""
Visualize.plot_ϕ_over_time!(args...; kwargs...) =
    _plot_ϕ_over_time(constrainedparamsanditersortime!, args...; kwargs...)

"""
    _plot_ϕ_over_time(plot_fn, args...; kwargs...)

Helper function to plot constrained parameter over iterations.

The arguments one can expect is `fig_or_ax`, `ekp`, `prior`, and `dim_idx` or
the last three arguments.
"""
function _plot_ϕ_over_time(plot_fn, args...; kwargs...)
    ekp, prior, dim_idx = last(args, 3)
    times = get_algorithm_time(ekp)
    pushfirst!(times, zero(eltype(times)))
    plot = plot_fn(args...; xvals = times, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    _modify_axis!(
        ax;
        xlabel = "Time",
        title = "ϕ over time [dim $(_get_dim_of_dist(prior, dim_idx))]",
        ylabel = "$(_get_prior_name(prior, dim_idx))",
    )
    return plot
end

"""
    Makie.plot!(plot::ConstrainedParamsAndItersOrTime)

Plot the constrained parameter versus iterations or time.
"""
function Makie.plot!(plot::ConstrainedParamsAndItersOrTime)
    ekp, prior, dim_idx, xvals = plot.ekp[], plot.prior[], plot.dim_idx[], plot.xvals[]
    ϕ = get_ϕ(prior, ekp)
    ensemble_size = size(first(ϕ))[2]
    nums = vcat((xval * ones(ensemble_size) for xval in xvals)...)
    param_nums = vcat((ϕ[idx][dim_idx, :] for idx in eachindex(xvals))...)
    valid_attributes = Makie.shared_attributes(plot, Makie.Scatter)
    Makie.scatter!(plot, nums, param_nums; valid_attributes...)
    return plot
end

# Define plot recipe for iteration or time vs constrained parameters scatter
# plots
# Define the function constrainedmeanparamsanditersortime whose functionality is
# implemented by specializing
# Makie.plot!(plot::ConstrainedMeanParamsAndItersOrTime)
@recipe(ConstrainedMeanParamsAndItersOrTime, ekp, prior, dim_idx) do scene
    Attributes(linewidth = 4.5)
end

"""
    Visualize.plot_ϕ_mean_over_iters!(axis, ekp, prior, dim_idx; plot_std = false, kwargs...)

Plot the mean constrained parameter of index `dim_idx` of `prior` against the
number of iterations on `axis`.

If `plot_std = true`, then the standard deviation of the constrained parameters
of the ensemble is also plotted.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines) and
(`Makie.Band`)[https://docs.makie.org/dev/reference/plots/band] if
`plot_std = true`.

Keyword arguments passed to `line_kwargs` and `band_kwargs` are merged with
`kwargs` when possible. The keyword arguments in `line_kwargs` and
`band_kwargs` take priority over the keyword arguments in `kwargs`.
"""
Visualize.plot_ϕ_mean_over_iters!(args...; plot_std = false, kwargs...) =
    _plot_ϕ_mean_over_iters(constrainedmeanparamsanditersortime!, args...; plot_std, kwargs...)

"""
    Visualize.plot_ϕ_mean_over_iters(gridposition, ekp, prior, dim_idx; plot_std = false, kwargs...)

Plot the mean constrained parameter of index `dim_idx` of `prior` against the
number of iterations on `gridposition`.

If `plot_std = true`, then the standard deviation of the constrained parameters
of the ensemble is also plotted.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines) and
(`Makie.Band`)[https://docs.makie.org/dev/reference/plots/band] if
`plot_std = true`.

Keyword arguments passed to `line_kwargs` and `band_kwargs` are merged with
`kwargs` when possible. The keyword arguments in `line_kwargs` and
`band_kwargs` take priority over the keyword arguments in `kwargs`.
"""
Visualize.plot_ϕ_mean_over_iters(args...; plot_std = false, kwargs...) =
    _plot_ϕ_mean_over_iters(constrainedmeanparamsanditersortime, args...; plot_std, kwargs...)

"""
    _plot_ϕ_mean_over_iters(plot_fn, args...; plot_std = false, kwargs...)

Helper function to plot mean constrained parameter against the number of
iterations.

The arguments one can expect is `fig_or_ax`, `ekp`, `prior`, and `dim_idx` or
the last three arguments.
"""
function _plot_ϕ_mean_over_iters(plot_fn, args...; plot_std, kwargs...)
    ekp, prior, dim_idx = last(args, 3)
    iters = collect(0:get_N_iterations(ekp))
    plot = plot_fn(args...; plot_std, xvals = iters, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    if plot_std
        axis_attribs = (
            xlabel = "Iterations",
            title = "Mean and std of ϕ over iterations [dim $(_get_dim_of_dist(prior, dim_idx))]",
            ylabel = "$(_get_prior_name(prior, dim_idx))",
            xticks = 0:get_N_iterations(ekp),
        )
    else
        axis_attribs = (
            xlabel = "Iterations",
            title = "Mean ϕ over iterations [dim $(_get_dim_of_dist(prior, dim_idx))]",
            ylabel = "$(_get_prior_name(prior, dim_idx))",
            xticks = 0:get_N_iterations(ekp),
        )
    end
    _modify_axis!(ax; axis_attribs...)
    return plot
end

"""
    Visualize.plot_ϕ_mean_over_time!(axis, ekp, prior, dim_idx; plot_std = false, kwargs...)

Plot the mean constrained parameter of index `dim_idx` of `prior` against time
on `axis`.

If `plot_std = true`, then the standard deviation of the constrained parameter
of the ensemble is also plotted.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines) and
(`Makie.Band`)[https://docs.makie.org/dev/reference/plots/band] if
`plot_std = true`.

Keyword arguments passed to `line_kwargs` and `band_kwargs` are merged with
`kwargs` when possible. The keyword arguments in `line_kwargs` and
`band_kwargs` take priority over the keyword arguments in `kwargs`.
"""
Visualize.plot_ϕ_mean_over_time!(args...; plot_std = false, kwargs...) =
    _plot_ϕ_mean_over_time(constrainedmeanparamsanditersortime!, args...; plot_std, kwargs...)

"""
    Visualize.plot_ϕ_mean_over_time(gridposition, ekp, prior, dim_idx; plot_std = false, kwargs...)

Plot the mean constrained parameter of index `dim_idx` of `prior` against time
on `gridposition`.

If `plot_std = true`, then the standard deviation of the constrained parameter
of the ensemble is also plotted.

Any keyword arguments is passed to the plotting function which takes in any
keyword arguments supported by
[`Makie.Lines`](https://docs.makie.org/dev/reference/plots/lines) and
(`Makie.Band`)[https://docs.makie.org/dev/reference/plots/band] if
`plot_std = true`.

Keyword arguments passed to `line_kwargs` and `band_kwargs` are merged with
`kwargs` when possible. The keyword arguments in `line_kwargs` and
`band_kwargs` take priority over the keyword arguments in `kwargs`.
"""
Visualize.plot_ϕ_mean_over_time(args...; plot_std = false, kwargs...) =
    _plot_ϕ_mean_over_time(constrainedmeanparamsanditersortime, args...; plot_std, kwargs...)

"""
    _plot_ϕ_mean_over_time(plot_fn, fig_or_ax, ekp, prior, dim_idx; kwargs...)

Helper function to plot mean constrained parameters against time.

The arguments one can expect is `fig_or_ax`, `ekp`, `prior`, and `dim_idx` or
the last three arguments.
"""
function _plot_ϕ_mean_over_time(plot_fn, args...; plot_std, kwargs...)
    ekp, prior, dim_idx = last(args, 3)
    times = accumulate(+, get_Δt(ekp))
    pushfirst!(times, zero(eltype(times)))
    plot = plot_fn(args...; plot_std, xvals = times, kwargs...)

    # If an axis is passed in, it must be the first argument. Otherwise, get the
    # current axis as determined by Makie. Note that current_axis() does not work
    # if an axis is passed in.
    ax = typeof(first(args)) <: Makie.AbstractAxis ? first(args) : Makie.current_axis()
    if plot_std
        axis_attribs = (
            xlabel = "Time",
            title = "Mean and std of ϕ over time [dim $(_get_dim_of_dist(prior, dim_idx))]",
            ylabel = "$(_get_prior_name(prior, dim_idx))",
        )
    else
        axis_attribs = (
            xlabel = "Time",
            title = "Mean ϕ over time [dim $(_get_dim_of_dist(prior, dim_idx))]",
            ylabel = "$(_get_prior_name(prior, dim_idx))",
        )
    end
    _modify_axis!(ax; axis_attribs...)
    return plot
end

"""
    Makie.plot!(plot::ConstrainedMeanParamsAndItersOrTime)

Plot the mean constrained parameter versus iterations or time.
"""
function Makie.plot!(plot::ConstrainedMeanParamsAndItersOrTime)
    ekp, prior, dim_idx, xvals = plot.ekp[], plot.prior[], plot.dim_idx[], plot.xvals[]
    ϕ_mean_vals = [get_ϕ_mean(prior, ekp, iter)[dim_idx] for iter in 1:length(xvals)]

    valid_attributes = Makie.shared_attributes(plot, Makie.Lines)
    hasproperty(plot, :line_kwargs) && (valid_attributes = merge(plot.line_kwargs, valid_attributes))
    Makie.lines!(plot, xvals, ϕ_mean_vals; valid_attributes...)

    if plot.plot_std[]
        stds = [std(get_ϕ(prior, ekp, iter)[dim_idx, :]) for iter in 1:length(xvals)]

        valid_attributes = Makie.shared_attributes(plot, Makie.Band)
        hasproperty(plot, :band_kwargs) && (valid_attributes = merge(plot.band_kwargs, valid_attributes))
        Makie.band!(plot, xvals, ϕ_mean_vals .- stds, ϕ_mean_vals .+ stds; valid_attributes...)
    end
    return plot
end

"""
    Visualize.plot_ϕ_over_iters(gridpositions, ekp, prior, name; kwargs...)

Plot the constrained parameter belonging to distribution `name` against the
number of iterations on the iterable `gridpositions`.

Any keyword arguments are passed to all plots.
"""
Visualize.plot_ϕ_over_iters(
    gridpositions,
    ekp::EnsembleKalmanProcess,
    prior::ParameterDistribution,
    name::AbstractString;
    kwargs...,
) = _plot_ϕ(Visualize.plot_ϕ_over_iters, gridpositions, ekp, prior, name; kwargs...)

"""
    Visualize.plot_ϕ_over_time(gridpositions, ekp, prior, name; kwargs...)

Plot the constrained parameter belonging to distribution `name` against time on
the iterable `gridpositions`.

Any keyword arguments are passed to all plots.
"""
Visualize.plot_ϕ_over_time(
    gridpositions,
    ekp::EnsembleKalmanProcess,
    prior::ParameterDistribution,
    name::AbstractString;
    kwargs...,
) = _plot_ϕ(Visualize.plot_ϕ_over_time, gridpositions, ekp, prior, name; kwargs...)

"""
    Visualize.plot_ϕ_mean_over_iters(gridpositions, ekp::EnsembleKalmanProcess, prior, name; kwargs...)

Plot the mean constrained parameter belonging to distribution `name` against the
number of iterations on the iterable `gridpositions`.

Any keyword arguments are passed to all plots.
"""
Visualize.plot_ϕ_mean_over_iters(
    gridpositions,
    ekp::EnsembleKalmanProcess,
    prior::ParameterDistribution,
    name::AbstractString;
    kwargs...,
) = _plot_ϕ(Visualize.plot_ϕ_mean_over_iters, gridpositions, ekp, prior, name; kwargs...)

"""
    Visualize.plot_ϕ_mean_over_time(gridpositions, ekp, prior, name; kwargs...)

Plot the mean constrained parameter belonging to distribution `name` against
time on the iterable `gridpositions`.

Any keyword arguments are passed to all plots.
"""
Visualize.plot_ϕ_mean_over_time(
    gridpositions,
    ekp::EnsembleKalmanProcess,
    prior::ParameterDistribution,
    name::AbstractString;
    kwargs...,
) = _plot_ϕ(Visualize.plot_ϕ_mean_over_time, gridpositions, ekp, prior, name; kwargs...)

"""
    _plot_ϕ(plot_fn, gridpositions, ekp, prior, name; kwargs...)

Helper function to plot `ϕ`-related quantities by the name of the distribution.
"""
function _plot_ϕ(plot_fn, gridpositions, ekp, prior, name::AbstractString; kwargs...)
    batch_idx = findfirst(x -> x == name, prior.name)
    isnothing(batch_idx) &&
        error("Cannot find a distribution with the name $name; the available distributions are $(prior.name)")
    dim_indices = batch(prior)[batch_idx]
    length(dim_indices) == length(gridpositions) || error(
        "The $name distribution has $(length(dim_indices)) dimensions, but $(length(gridpositions)) gridpositions are passed in",
    )
    local plot
    for (gridposition, dim_idx) in zip(gridpositions, dim_indices)
        plot = plot_fn(gridposition, ekp, prior, dim_idx; kwargs...)
    end
    return plot
end

"""
    _modify_axis!(ax; kwargs...)

Modify an axis in place.

This currently supports only `:title`, `:xlabel`, `:ylabel`, and `:xticks`.

This function is hacky as Makie does not currently have support for axis hints,
so this is a workaround which will work in most cases.

For example, it is not possible to tell if the user want an empty title or not
since this function assumes that if the default is used, then it is okay to
overwrite.
"""
function _modify_axis!(ax; kwargs...)
    defaults = Dict(:title => "", :xlabel => "", :ylabel => "", :xticks => Makie.Automatic())
    for (attrib, val) in kwargs
        if attrib in keys(defaults) && getproperty(ax, attrib)[] == defaults[attrib]
            getproperty(ax, attrib)[] = val
        end
    end
    return nothing
end

"""
    _get_dim_of_dist(prior, dim_idx)

Get the dimension of the distribution belonging to `dim_idx` in `prior`.
"""
function _get_dim_of_dist(prior, dim_idx)
    for param_arr in batch(prior)
        if dim_idx in param_arr
            return findfirst(x -> x == dim_idx, param_arr)
        end
    end
    error("Could not find name corresponding to parameter $dim_idx")
end

"""
    _get_prior_name(prior, dim_idx)

Get name of `dim_idx` parameter from `prior`.
"""
function _get_prior_name(prior, dim_idx)
    for (i, param_arr) in enumerate(batch(prior))
        if dim_idx in param_arr
            return get_name(prior)[i]
        end
    end
    error("Could not find name corresponding to parameter $dim_idx")
end
"""
    _find_appropriate_error(ekp, error_name)

Find the appropriate name of the error to return if `error_name` is `nothing`.
Otherwise, return `error_name`.
"""
function _find_appropriate_error(ekp, error_name)
    if isnothing(error_name)
        process = get_process(ekp)
        prior_mean = get_prior_mean(process)
        prior_cov = get_prior_cov(process)
        error_name = (isnothing(prior_mean) || isnothing(prior_cov)) ? "loss" : "bayes_loss"
    end
    return error_name
end

end
