# Observations

The Observations object facilitates convenient storing, grouping and minibatching over observations.

## Summary

### The key objects
1. The `Observation` is a container for an observed variables ("samples"), their noise covariances ("covariances"), and names ("names"). They are easily stackable to help build larger heterogenous observations
2. The `Minibatcher` facilitate data streaming (minibatching), where a user can submit large group of observations, that are then batched up and looped over in epochs.
3. The `ObservationSeries` contains the list of `Observation`s and `Minibatcher` and the utilities to get the current batch etc.

!!! note "But I just pass in a vector of data and a covariance - Is that OK?"
    Users can indeed set up an experiment with just one data sample and covariance matrix for the noise. However internally these are still stored as an `ObservationSeries` with a special minibatcher that does nothing (created by `no_minibatcher(size)`). 

### Recommended constructor: A single stacked observation

Here the user has data for two independent variables: the five-dimensional `y` and the twenty-dimensional `z`. The observations of `y` are all independent, while the observations of `z` have some structure.

We recommend users build using a dictionary,
```@example
using EnsembleKalmanProcesses, for `Observation`
using LinearAlgebra # for `I`

# observe variable y with some diagonal noise
y = ones(5)
cov_y = 0.01*I

# observe variable z with some tri-diagonal noise
z = zeros(20)
cov_z = Tridiagonal(0.1*ones(19), ones(20), 0.1*ones(19))

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

# getting information out
get_obs(full_obs) # returns [y,z]
get_obs_noise_cov(full_obs) # returns block-diagonal matrix with blocks [cov_y 0; 0 cov_z]

# getters `get_*` can be used for the internally stored information too e.g. `get_names(full_obs)`
```

### Recommended constructor: Many stacked observations

Imagine the user has 1000 independent data samples  for two independent variables above.
Rather than stacking all the data together at once (forming a full system of size `1000*20*5` to update at each step) instead the user wishes to stream the data and do updates with random batches of 20 observations at each iteration.

```@example
# given a vector of 1000 `Observation` called thousand_full_obs
using EnsembleKalmanProcesses # for `RandomFixedSizeMinibatcher`, `ObservationSeries`

minibatcher = RandomFixedSizeMinibatcher(20) # batches the epoch of size 1000, into batches of size 20

observation_series = ObservationSeries(
    Dict(
        "observations" => thousand_full_obs,
        "minibatcher" => minibatcher,
    ),
)

# some example methods to get information out at the current minibatch:
get_current_minibatch(observation_series) # returns [i₁, i₂, ..., i₂₀],  the current minibatch subset of indices 1:1000
get_obs(observation_series) # returns [yi₁, zi₁, yi₂, zi₂,..., yi₂₀, zi₂₀], the data sample for the current minibatch
get_obs_noise_cov(observation_series) # returns block-diagonal matrix with blocks [cov_yi₁  0 ... 0 ; 0 cov_zi₁ 0 ... 0; ... ; 0 ... 0 cov_yi₂₀ 0; 0 ... 0 cov_zi₂₀]
```

## Minibatchers

Minibatchers are modular and must be derived from the `Minibatcher` `abstract type`. They contain a method `create_new_epoch!(minibatcher,args...;kwargs)` that creates a sampling of an epoch of data. For example, if we have 1000 data observations, the epoch is `1:1000`, and a typical minibatching is a random partitioning of `1:1000` into a batch-size (e.g., 20) leading to 50 minibatches.  

Some of the implemented Minibatchers
- `FixedMinibatcher(given_batches, "order")`, (default `method = "order"`), minibatches are fixed and run through in order for all epochs
- `FixedMinibatcher(given_batches, "random")`, minibatches are fixed, but are run through in a randomly chosen order in each epoch
- `no_minibatcher(size)`, creates a `FixedMinibatcher` with just one batch which is the epoch (effectively no minibatching)
- `RandomFixedSizeMinibatcher(minibatch_size, "trim")`, (default `method = "trim"`) creates minibatches of size `minibatch_size` by randomly sampling the epoch, if the minibatch size does not divide into the number of samples it will ignore the remainder (and thus preserving a constant batch size)
- `RandomFixedSizeMinibatcher(minibatch_size, "extend")`, creates minibatches of size `minibatch_size` by randomly sampling the epoch, if the minibatch size does not divide into the number of samples it will include the remainder in the final batch (and thus will cover the entirety of the data, with a larger final batch)

## Observation Series

One can additionally provide a vector of `names` to name each `Observation` in the `ObservationSeries` by giving using the `Dict` entry `"names" => names`


