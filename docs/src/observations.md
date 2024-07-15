# Observations

The Observations object facilitates convenient storing, grouping and minibatching over observations.

## The key objects
1. The `Observation` is a container for an observed variables ("samples"), their noise covariances ("covariances"), and names ("names"). They are easily stackable to help build larger heterogenous observations
2. The `Minibatcher` facilitate data streaming (minibatching), where a user can submit large group of observations, that are then batched up and looped over in epochs.
3. The `ObservationSeries` contains the list of `Observation`s and `Minibatcher` and the utilities to get the current batch etc.

!!! note "I usually just pass in a vector of data and a covariance to EKP"
    Users can indeed set up an experiment with just one data sample and covariance matrix for the noise. However internally these are still stored as an `ObservationSeries` with a special minibatcher that does nothing (created by `no_minibatcher(size)`). 

## Recommended constructor: A single (stacked) observation

Here the user has data for two independent variables: the five-dimensional `y` and the eight-dimensional `z`. The observations of `y` are all independent, while the observations of `z` have some structure.

We recommend users build an `Observation` using the `Dict` constructor and make use of the `combine_observations()` utility.
```@example ex1
using EnsembleKalmanProcesses # for `Observation`
using LinearAlgebra # for `I`, `Tridiagonal`

# observe variable y with some diagonal noise
y = ones(5)
cov_y = 0.01*I

# observe variable z with some tri-diagonal noise
zdim = 8
z = zeros(zdim)
cov_z = Tridiagonal(0.1*ones(zdim-1), ones(zdim), 0.1*ones(zdim-1))

y_obs = Observation(
    Dict(
        "samples" => y,
        "covariances" => cov_y,
        "names" => "y",
    ),
)

z_obs = Observation(
    Dict(
        "samples" => z,
        "covariances" => cov_z,
        "names" => "z",
    ),
)

full_obs = combine_observations([y_obs,z_obs]) # conveniently stacks the observations
```

```@example ex1
# getting information out
get_obs(full_obs) # returns [y,z]
```
```@example ex1
get_obs_noise_cov(full_obs) # returns block-diagonal matrix with blocks [cov_y 0; 0 cov_z]
```
getters `get_*` can be used for the internally stored information too including:
``` @example ex1
get_names(full_obs)
```
There are some other fields stored such as indices of the `y` and `z` components
```@example ex1
get_indices(full_obs)
```

## Recommended constructor: Many stacked observations

Imagine the user has 100 independent data samples  for two independent variables above, where the `k`th `y` sample is = `k*ones(5)` for each `k=1:100`.
Rather than stacking all the data together at once (forming a full system of size `100*(8+5)` to update at each step) instead the user wishes to stream the data and do updates with random batches of 5 observations at each iteration.

!!! note "Why would I choose to minibatch?"
    The memory- and time-scaling of many EKP methods is worse-than-linear in the observation dimension, therefore there is often large computational benefit to minibatch EKP updates. Such costs must be weighed against the cost of additional forward map evaluations needed to minibatching over one full epoch. 

```@setup ex2
using EnsembleKalmanProcesses
using LinearAlgebra

hundred_full_obs = []
for k = 1:100
    y = k*ones(5)
    cov_y = 0.01*I(5)

    z = zeros(8)
    cov_z = Tridiagonal(0.1*ones(7), ones(8), 0.1*ones(7))

    y_obs = Observation(
          Dict(
        "samples" => y,
        "covariances" => cov_y,
        "names" => "y_$k",
    ),
)

z_obs = Observation(
    Dict(
        "samples" => z,
        "covariances" => cov_z,
        "names" => "z_$k",
    ),
)
push!(hundred_full_obs, combine_observations([y_obs,z_obs]))
end
T = promote_type((typeof(h) for h in hundred_full_obs)...)
hundred_full_obs = [convert(T, h) for h in hundred_full_obs]
```

```@example ex2
# given a vector of 100 `Observation`s called hundred_full_obs, where `y_k = k*y`
using EnsembleKalmanProcesses # for `RandomFixedSizeMinibatcher`, `ObservationSeries`, `Minibatcher`

minibatcher = RandomFixedSizeMinibatcher(5) # batches the epoch of size 100, into batches of size 5

observation_series = ObservationSeries(
    Dict(
        "observations" => hundred_full_obs,
        "minibatcher" => minibatcher,
    ),
)
```

```@example ex2
# some example methods to get information out at the current minibatch:
get_current_minibatch(observation_series) # returns [i₁, ..., i₅],  the current minibatch subset of indices 1:100
```

```@example ex2
get_obs(observation_series) # returns [yi₁, zi₁, ..., yi₅, zi₅], the data sample for the current minibatch
```

```@example ex2
get_obs_noise_cov(observation_series) # returns block-diagonal matrix with blocks [cov_yi₁  0 ... 0 ; 0 cov_zi₁ 0 ... 0; ... ; 0 ... 0 cov_yi₅ 0; 0 ... 0 cov_zi₅]
```

minibatches are updated internally to the `update_ensemble!(ekp,...)` step via a call to
```@example ex2
update_minibatch!(observation_series)
get_current_minibatch(observation_series)
```



## Minibatchers

Minibatchers are modular and must be derived from the `Minibatcher` `abstract type`. They contain a method `create_new_epoch!(minibatcher,args...;kwargs)` that creates a sampling of an epoch of data. For example, if we have 100 data observations, the epoch is `1:100`, and one possible minibatching is a random partitioning of `1:100` into a batch-size (e.g., 5) leading to 20 minibatches.  

Some of the implemented Minibatchers
- `FixedMinibatcher(given_batches, "order")`, (default `method = "order"`), minibatches are fixed and run through in order for all epochs
- `FixedMinibatcher(given_batches, "random")`, minibatches are fixed, but are run through in a randomly chosen order in each epoch
- `no_minibatcher(size)`, creates a `FixedMinibatcher` with just one batch which is the epoch (effectively no minibatching)
- `RandomFixedSizeMinibatcher(minibatch_size, "trim")`, (default `method = "trim"`) creates minibatches of size `minibatch_size` by randomly sampling the epoch, if the minibatch size does not divide into the number of samples it will ignore the remainder (and thus preserving a constant batch size)
- `RandomFixedSizeMinibatcher(minibatch_size, "extend")`, creates minibatches of size `minibatch_size` by randomly sampling the epoch, if the minibatch size does not divide into the number of samples it will include the remainder in the final batch (and thus will cover the entirety of the data, with a larger final batch)

## Identifiers

One can additionally provide a vector of `names` to name each `Observation` in the `ObservationSeries` by giving using the `Dict` entry `"names" => names`.

To think about the differences between the identifiers for `Observation` and `ObservationSeries` consider an application of observing the average state of a dynamical system over 100 time windows. The time windows will be batched over during the calibration.

The compiled information is given in the object:
```julia
yz_observation_series::ObservationSeries
```
As this contains many time windows, setting the names of the `ObservationSeries` objects to index the time window is a sensible identifier, for example,
```julia
get_names(yz_observation_series)
> ["window_1", "window_2", ..., "window_100"]
```
The individual `Observation`s should refer only to the state being measured, sosuitable identifiers might be, for example,
```julia
obs = get_observations(yz_observation_series)[1] # get first observation in the series
get_names(obs)
> ["y_window_average", "z_window_average"]
```

