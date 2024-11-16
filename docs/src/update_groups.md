# [Update Groups] (@id update-groups)

The `UpdateGroup` object facilitates blocked EKP updates, based on a provided updating a series of blocks of paired (parameters and data). As many of the `Process` updates can scale quadraticly in the data dimension, this reduces computational complexity of the update.

##  Recommended construction

The key component to construct update groups starts with constructing the prior, and the observations. Parameter distributions and observations may be constructed in units and given names, and these names are utilized to build the update groups with a convenient constructor.

For example, we take code snippets from the example found in `examples/UpdateGroups/calibrate.jl`. This example is concerned with learning several parameters in a coupled Lorenz 96 multiscale system, by gathering data from moments from the fast `Y` and slow `X` moments.

In this exmaple the parameter distribution we create a prior from several *named* distributions.
```julia
param_names = ["F", "G", "h", "c", "b"]

prior_F = ParameterDistribution(
    Dict(
        "name" => param_names[1],
        "distribution" => Parameterized(MvNormal([1.0, 0.0, -2.0], I)),
        "constraint" => repeat([bounded_below(0)], 3),
    ),
) # gives 3-D dist
prior_G = constrained_gaussian(param_names[2], 5.0, 4.0, 0, Inf)
prior_h = constrained_gaussian(param_names[3], 5.0, 4.0, 0, Inf)
prior_c = constrained_gaussian(param_names[4], 5.0, 4.0, 0, Inf)
prior_b = constrained_gaussian(param_names[5], 5.0, 4.0, 0, Inf)
priors = combine_distributions([prior_F, prior_G, prior_h, prior_c, prior_b])
```
Now we likewise construct blocked observations from several *named* observations
```julia
# given a list of vector statistics y and their covariances Γ 
data_block_names = ["<X>", "<Y>", "<X^2>", "<Y^2>", "<XY>"]

observation_vec = []
for i in 1:length(data_block_names)
    push!(
        observation_vec,
        Observation(Dict("samples" => y[i], "covariances" => Γ[i], "names" => data_block_names[i])),
    )
end
observation = combine_observations(observation_vec)
```
Now we define the update groups of our choice as a dictionary, this is the critical feature of interest here. In this case we create two blocks (though one could create other updates here)
```julia
# update parameters F,G with data <X>,<X^2>
# update parameters h, c, b with data <Y>, <Y^2>, <XY>
group_identifiers = Dict(
    ["F", "G"] => ["<X>", "<X^2>"],
    ["h", "c", "b"] => ["<Y>", "<Y^2>", "<XY>"],
)
```
We then create the update groups with our constructor
```julia
update_groups = create_update_groups(prior, observation, group_identifiers)
```
and this can then be entered into the `EnsembleKalmanProcess` object as a keyword argument
```julia
# ... initial_params = construct_initial_ensemble(rng, priors, N_ens)
ekiobj = EnsembleKalmanProcess(initial_params, observation, Inversion(), update_groups = update_groups)
```

## Advice for constructing blocks
1. The blocks cannot contain repeated parameters (i.e. parameters cannot be updated "twice")
2. Block structure is user-defined, and directly assumes that there is no correlation between blocks. It is up to the user to confirm if there truly is independence between different blocks. Otherwise convergence properties may suffer.
3. This can be used in conjunction with minibatching, so long as the defined data objects are available in all data observations.

## What happens internally?

We simply perform an independent `update_ensemble!` for each provided pairing and combine model output and updated parameters afterwards. Note that even without specifying an update group, by default EKP will always be construct one under-the-hood.

!!! note "In future..."
    In theory this opens up the possibility to have different configurations, or even processes, in different groups. This could be useful when parameter-data pairings are highly heterogeneous and so the user may wish to exploit, for example, the different processes scaling properties. However this has not been implemented.