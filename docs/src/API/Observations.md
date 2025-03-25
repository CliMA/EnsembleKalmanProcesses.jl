# Observations

```@meta
CurrentModule = EnsembleKalmanProcesses
```
## Observation
```@docs
Observation
get_samples
get_covs
get_inv_covs
get_names
get_indices
combine_observations
get_obs(o::OB) where {OB <: Observation}
get_obs_noise_cov(o::OB) where {OB <: Observation}
get_obs_noise_cov_inv(o::OB) where {OB <: Observation}
```
## Covariance utilities
```@docs
tsvd_mat
tsvd_cov_from_samples
SVDplusD
DminusTall
```

## Minibatcher
```@docs
FixedMinibatcher
no_minibatcher
create_new_epoch!(m::FM) where {FM <: FixedMinibatcher}
get_minibatches(m::FM) where {FM <: FixedMinibatcher}
get_method(m::FM) where {FM <: FixedMinibatcher}
get_rng(m::FM) where {FM <: FixedMinibatcher}
```

```@docs
RandomFixedSizeMinibatcher
get_minibatch_size(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher}
get_method(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher}
get_rng(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher}
get_minibatches(m::RFSM) where {RFSM <: RandomFixedSizeMinibatcher}
```
## ObservationSeries
```@docs
ObservationSeries
get_observations(os::OS) where {OS <: ObservationSeries}
get_minibatches(os::OS) where {OS <: ObservationSeries}
get_current_minibatch_index(os::OS) where {OS <: ObservationSeries}
get_minibatcher(os::OS) where {OS <: ObservationSeries}
update_minibatch!(os::OS) where {OS <: ObservationSeries}
get_current_minibatch(os::OS) where {OS <: ObservationSeries}
get_obs(os::OS) where {OS <: ObservationSeries}
get_obs_noise_cov(os::OS) where {OS <: ObservationSeries}
get_obs_noise_cov_inv(os::OS) where {OS <: ObservationSeries}
```