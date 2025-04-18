module EnsembleKalmanProcessesMakieExt

using Makie

import EnsembleKalmanProcesses: EnsembleKalmanProcess, get_N_iterations, get_Δt, get_error, get_ϕ
import EnsembleKalmanProcesses.Visualize as Visualize
using EnsembleKalmanProcesses.ParameterDistributions

using Random

# Plotting recipe do not support plotting subplots, so this is a convience wrapper that
# plot the priors with no support for passing in keyword arguments
"""
    Visualize.plot_priors(fig::Union{Makie.Figure, Makie.GridLayout, Makie.GridPosition, Makie.GridSubposition},
                           pd::ParameterDistribution;
                           constrained = true,
                           n_sample = 1e4,
                           rng = Random.GLOBAL_RNG)

Plot the prior distributions `pd` on `fig`.
"""
function Visualize.plot_priors(
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
    Visualize.plot_priors(fig::Union{Makie.Figure, Makie.GridLayout},
                           pd::PDT;
                           constrained = true,
                           n_sample = 1e4,
                           rng = Random.GLOBAL_RNG)
                           where {PDT <: ParameterDistributionType}

Plot the distribution on `fig`.
"""
# TODO: Update this function and make sure it is the same as PlotRecipes
function Visualize.plot_priors(
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

# Define plot recipe for iterations vs errors line plots
@recipe(ErrorAndItersOrTime, ekp, xvals) do scene
    Theme()
end

"""
    Visualize.plot_error_over_iters(figure, ekp)

Plot the errors from `ekp` against the number of iterations on `figure`.
"""
Visualize.plot_error_over_iters(fig, ekp; kw_args...) =
    _plot_error_over_iters(erroranditersortime, fig, ekp; kw_args...)

"""
    Visualize.plot_error_over_iters!(axis, ekp)

Plot the errors from `ekp` against the number of iterations on `axis`.
"""
Visualize.plot_error_over_iters!(ax, ekp; kw_args...) =
    _plot_error_over_iters(erroranditersortime!, ax, ekp; kw_args...)

"""
    _plot_errors_over_iters(plot_fn, fig_or_ax, ekp; kw_args...)

Helper function to plot the errors against the number of iterations.
"""
function _plot_error_over_iters(plot_fn, fig_or_ax, ekp; kw_args...)
    iters = 1:get_N_iterations(ekp) # off by one error here?, need to check with example and calibration being ran
    plot = plot_fn(fig_or_ax, ekp, iters; kw_args...)
    ax = typeof(fig_or_ax) <: Makie.AbstractAxis ? fig_or_ax : Makie.current_axis()
    _modify_axis!(ax; xlabel = "Iterations", title = "Error over iterations", ylabel = "Error")
    return plot
end

"""
    Visualize.plot_error_over_time(figure, ekp; kw_args...)

Plot the errors from `ekp` against time on `figure`.
"""
Visualize.plot_error_over_time(fig, ekp; kw_args...) = _plot_error_over_time(erroranditersortime, fig, ekp; kw_args...)

"""
    Visualize.plot_error_over_time!(axis, ekp; kw_args...)

Plot the errors from `ekp` against time on `axis`.
"""
Visualize.plot_error_over_time!(ax, ekp; kw_args...) = _plot_error_over_time(erroranditersortime!, ax, ekp; kw_args...)

"""
    _plot_error_over_time(plot_fn, fig_or_ax, ekp; kw_args...)

Helper function to plot errors over time.
"""
function _plot_error_over_time(plot_fn, fig_or_ax, ekp; kw_args...)
    times = accumulate(+, get_Δt(ekp))
    plot = plot_fn(fig_or_ax, ekp, times; kw_args...)
    ax = typeof(fig_or_ax) <: Makie.AbstractAxis ? fig_or_ax : Makie.current_axis()
    _modify_axis!(ax; xlabel = "Time", title = "Error over time", ylabel = "Error")
    return plot
end

"""
    Makie.plot!(plot::ErrorAndItersOrTime)

Plot the errors against iteration or time.
"""
function Makie.plot!(plot::ErrorAndItersOrTime)
    xvals = plot.xvals[]
    errors = get_error(plot.ekp[])
    valid_attributes = Makie.shared_attributes(plot, Makie.Lines)
    Makie.lines!(plot, xvals, errors; valid_attributes...)
    return plot
end

# Define plot recipes for iteration or time vs constrained parameters scatter plots
@recipe(ConstrainedParamsAndItersOrTime, ekp, prior, param_idx, xvals) do scene
    Theme()
end

"""
    Visualize.plot_ϕ_over_iters(figure, ekp, prior, param_idx; plot_kwargs...)

Plot the constrained parameters of index `param_idx` against time on `figure`.
"""
Visualize.plot_ϕ_over_iters(fig, ekp, prior, param_idx; kw_args...) =
    _plot_ϕ_over_iters(constrainedparamsanditersortime, fig, ekp, prior, param_idx; kw_args...)

"""
    Visualize.plot_ϕ_over_iters!(axis, ekp, prior, param_idx; plot_kwargs...)

Plot the constrained parameters of index `param_idx` against time on `axis`.
"""
Visualize.plot_ϕ_over_iters!(ax, ekp, prior, param_idx; kw_args...) =
    _plot_ϕ_over_iters(constrainedparamsanditersortime!, ax, ekp, prior, param_idx; kw_args...)

"""
    _plot_ϕ_over_iters(plot_fn, fig_or_ax, ekp, prior, param_idx; kw_args...)

Helper function to plot constrained parameters against the number of iterations.
"""
function _plot_ϕ_over_iters(plot_fn, fig_or_ax, ekp, prior, param_idx; kw_args...)
    iters = collect(0:get_N_iterations(ekp))
    plot = plot_fn(fig_or_ax, ekp, prior, param_idx, iters; kw_args...)
    ax = typeof(fig_or_ax) <: Makie.AbstractAxis ? fig_or_ax : Makie.current_axis()
    _modify_axis!(
        ax;
        xlabel = "Iterations",
        title = "ϕ over iterations [dim $(_get_dim_of_dist(prior, param_idx))]",
        ylabel = "$(_get_prior_name(prior, param_idx))",
    )
    return plot
end

"""
    Visualize.plot_ϕ_over_iters(figure, ekp, prior, param_idx; plot_kwargs...)

Plot the constrained parameters of index `param_idx` against time on `figure`.
"""
Visualize.plot_ϕ_over_time(fig, ekp, prior, param_idx; kw_args...) =
    _plot_ϕ_over_time(constrainedparamsanditersortime, fig, ekp, prior, param_idx; kw_args...)

"""
    Visualize.plot_ϕ_over_time!(axis, ekp, prior, param_idx; plot_kwargs...)

Plot the constrained parameters of index `param_idx` against time on `axis`.
"""
Visualize.plot_ϕ_over_time!(ax, ekp, prior, param_idx; kw_args...) =
    _plot_ϕ_over_time(constrainedparamsanditersortime!, ax, ekp, prior, param_idx; kw_args...)

"""
    _plot_ϕ_over_time(plot_fn, fig_or_ax, ekp, prior, param_idx; kw_args...)

Helper function to plot constrained parameters over iterations.
"""
function _plot_ϕ_over_time(plot_fn, fig_or_ax, ekp, prior, param_idx; kw_args...)
    times = accumulate(+, get_Δt(ekp))
    plot = plot_fn(fig_or_ax, ekp, prior, param_idx, times; kw_args...)
    ax = typeof(fig_or_ax) <: Makie.AbstractAxis ? fig_or_ax : Makie.current_axis()
    _modify_axis!(
        ax;
        xlabel = "Time",
        title = "ϕ over time [dim $(_get_dim_of_dist(prior, param_idx))]",
        ylabel = "$(_get_prior_name(prior, param_idx))",
    )
    return plot
end

"""
    Makie.plot!(plot::ConstrainedParamsAndItersOrTime)

Plot the constrained parameters versus iterations or time.
"""
function Makie.plot!(plot::ConstrainedParamsAndItersOrTime)
    ekp, prior, param_idx, xvals = plot.ekp[], plot.prior[], plot.param_idx[], plot.xvals[]
    ϕ = get_ϕ(prior, ekp)
    ensemble_size = size(first(ϕ))[2]
    nums = vcat((xval * ones(ensemble_size) for xval in xvals)...)
    param_nums = vcat((ϕ[idx][param_idx, :] for idx in eachindex(xvals))...)
    valid_attributes = Makie.shared_attributes(plot, Makie.Scatter)
    Makie.scatter!(plot, nums, param_nums; valid_attributes...)
    return plot
end

"""
    _modify_axis!(ax; kwargs...)

Modify an axis in place.

This currently supports only `:title`, `:xlabel`, and `:ylabel`.

This function is hacky as Makie does not currently have support for axis hints.

Note that it is not possible to tell if the user want an empty title or not
since this function assumes that if the default is used, then it is fine to
overwrite.
"""
function _modify_axis!(ax; kwargs...)
    defaults = Dict(:title => "", :xlabel => "", :ylabel => "")
    for (attrib, val) in kwargs
        if attrib in keys(defaults) && getproperty(ax, attrib)[] == defaults[attrib]
            getproperty(ax, attrib)[] = val
        end
    end
    return nothing
end

"""
    _get_dim_of_dist(prior, param_idx)

Get the dimension of the distribution belonging to `param_idx` parameter.
"""
function _get_dim_of_dist(prior, param_idx)
    for param_arr in batch(prior)
        if param_idx in param_arr
            return findfirst(x -> x == param_idx, param_arr)
        end
    end
    error("Could not find name corresponding to parameter $param_idx")
end

"""
    _get_prior_name(prior, param_idx)

Get name of `param_idx` parameter from `prior`.
"""
function _get_prior_name(prior, param_idx)
    for (i, param_arr) in enumerate(batch(prior))
        if param_idx in param_arr
            return get_name(prior)[i]
        end
    end
    error("Could not find name corresponding to parameter $param_idx")
end

end
