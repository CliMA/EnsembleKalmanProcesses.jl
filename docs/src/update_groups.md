# [Update Groups] (@id update-groups)

The `UpdateGroup` object facilitates blocked EKP updates, based on a provided updating a series user-defined pairs of parameters and data. This allows users to enforce any *known* (in)dependences between different groups of parameters during the update. For example, 
```julia
# update parameter 1 with data 1 and 2
# update parameters 2 and 3 jointly with data 2, 3, and 4
Dict(
    ["parameter_1"] => ["data_1", "data_2"], 
    ["parameter_2", "parameter_3"] => ["data_2", "data_3", "data_4"], 
)
```
Construction and passing of this into the EnsembleKalmanProcesses is detailed below.

!!! note "This improves scaling at the cost of user-imposed structure"
    As many of the `Process` updates scale say with ``d^\alpha``, in the data dimension ``d`` and ``\alpha > 1`` (super-linearly),  update groups with ``K`` groups of equal size will improving this scaling to ``K (\frac{d}{K})^\alpha``.

##  Recommended construction - shown by example

The key component to construct update groups starts with constructing the prior and the observations. Parameter distributions and observations may be constructed in units and given names, and these names are utilized to build the update groups with a convenient constructor `create_update_groups`.

For illustration, we take code snippets from the example found [here](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/UpdateGroups/). This example is concerned with learning several parameters in a coupled two-scale Lorenz 96 system:
```math
\begin{aligned}
 \frac{\partial X_i}{\partial t} & = -X_{i-1}(X_{i-2} - X_{i+1}) - X_i - GY_i + F_1 + F_2\,\sin(2\pi t F_3)\\
 \frac{\partial Y_i}{\partial t} & = -cbY_{i+1}(Y_{i+2} - Y_{i-1}) - cY_i + \frac{hc}{b} X_i 
\end{aligned}
```
Parameters are learnt by fitting estimated moments of a realized `X` and `Y` system, to some target moments over a time interval.

We create a prior by combining several *named* `ParameterDistribution`s.
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
Now we likewise construct observed moments by combining several *named* `Observation`s
```julia
# given a list of vector statistics y and their covariances Γ 
data_block_names = ["<X>", "<Y>", "<X^2>", "<Y^2>", "<XY>"]

observation_vec = []
for i in 1:length(data_block_names)
    push!(
        observation_vec,
        Observation(Dict(
            "samples" => y[i],
            "covariances" => Γ[i],
            "names" => data_block_names[i]
        )),
    )
end
observation = combine_observations(observation_vec)
```
Finally, we are ready to define the update groups. We may specify our choice by partitioning the parameter names as keys of a dictionary, and their paired data names as values. Here we create two groups:
```julia
# update parameters F,G with data <X>, <X^2>, <XY>
# update parameters h, c, b with data <Y>, <Y^2>, <XY>
group_identifiers = Dict(
    ["F", "G"] => ["<X>", "<X^2>", "<XY>"],
    ["h", "c", "b"] => ["<Y>", "<Y^2>", "<XY>"],
)
```
We then create the update groups with our convenient constructor
```julia
update_groups = create_update_groups(prior, observation, group_identifiers)
```
and this can then be entered into the `EnsembleKalmanProcess` object as a keyword argument
```julia
# initial_params = construct_initial_ensemble(rng, priors, N_ens) 
ekiobj = EnsembleKalmanProcess(
    initial_params,
    observation,
    Inversion(),
    update_groups = update_groups
)
```

## What happens internally?

We simply perform an independent `update_ensemble!` for each provided pairing and combine model output and updated parameters afterwards. Note that even without specifying an update group, by default EKP will always be construct one under-the-hood.



## Advice for constructing blocks
1. A parameter cannot appear in more than one block (i.e. parameters cannot be updated more than once)
2. The block structure is user-defined, and directly assumes that there is no correlation between blocks. It is up to the user to confirm if there truly is independence between different blocks. Otherwise convergence properties may suffer.
3. This can be used in conjunction with minibatching, so long as the defined data objects are available in all `Observation`s in the series.

!!! note "In future..."
    In theory this opens up the possibility to have different configurations, or even processes, in different groups. This could be useful when parameter-data pairings are highly heterogeneous and so the user may wish to exploit, for example, the different processes scaling properties. However this has not yet been implemented.