module PlotRecipes

using RecipesBase
using Random
using ..ParameterDistributions

export plot_marginal_hist

@recipe function plot(pd::ParameterDistribution; constrained = true, n_sample = 1e4, rng = Random.GLOBAL_RNG)
    samples = sample(rng, pd, Int(n_sample))
    if constrained
        samples = transform_unconstrained_to_constrained(pd, samples)
    end

    # First attempt,  make it into a samples dist and plot histograms instead
    n_plots = ndims(pd)
    batches = batch(pd)

    rows = Int(ceil(sqrt(n_plots)))
    cols = Int(floor(sqrt(n_plots)))
    tfs = 16
    fs = 12

    # subplot attr
    legend := false
    framestyle := repeat([:axes], n_plots)
    grid := false

    layout := n_plots
    size --> (rows * 400, cols * 400)
    titlefontsize --> tfs
    xtickfontsize --> fs
    ytickfontsize --> fs
    xguidefontsize --> fs
    yguidefontsize --> fs

    for i in 1:n_plots
        batch_id = [j for j = 1:length(batches) if i ∈ batches[j]][1]
        dim_in_batch = i - minimum(batches[batch_id]) + 1 # i.e. if i=5 in batch 3:6, this would be "3"
        @series begin
            seriestype := :histogram
            normalize --> :pdf
            color := batch_id
            subplot := i
            title := pd.name[batch_id] * " (dim " * string(dim_in_batch) * ")"
            samples[i, :]
        end
    end

end

@recipe function plot(d::PDT; n_sample = 1e4, rng = Random.GLOBAL_RNG) where {PDT <: ParameterDistributionType}
    samples = sample(rng, d, Int(n_sample))

    # First attempt,  make it into a samples dist and plot histograms instead
    n_plots = ndims(d)

    size_l = Int(ceil(sqrt(n_plots)))
    tfs = 16
    fs = 12

    # subplot attr
    legend := false
    framestyle := repeat([:axes], n_plots)
    grid := false
    layout := n_plots
    size --> (size_l * 400, size_l * 400)
    titlefontsize --> tfs
    xtickfontsize --> fs
    ytickfontsize --> fs
    xguidefontsize --> fs
    yguidefontsize --> fs

    for i in 1:n_plots
        @series begin
            seriestype := :histogram
            normalize --> :pdf
            subplot := i
            title := "dim " * string(i)
            samples[i, :]
        end
    end

end



end #module
