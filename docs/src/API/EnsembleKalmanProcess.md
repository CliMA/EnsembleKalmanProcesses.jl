# EnsembleKalmanProcess

```@meta
CurrentModule = EnsembleKalmanProcesses
```

## Primary objects and functions
```@docs
EnsembleKalmanProcess
construct_initial_ensemble(rng::AbstractRNG, prior::ParameterDistribution, N_ens)
update_ensemble!
```
## Getter functions

```@docs
get_u
get_g
get_ϕ
get_obs(ekp::EnsembleKalmanProcess)
get_obs(ekp::EnsembleKalmanProcess, iteration)
get_obs_noise_cov(ekp::EnsembleKalmanProcess)
get_obs_noise_cov(ekp::EnsembleKalmanProcess, iteration)
get_obs_noise_cov_inv(ekp::EnsembleKalmanProcess)
get_obs_noise_cov_inv(ekp::EnsembleKalmanProcess, iteration)
get_u_final
get_g_final
get_ϕ_final
get_u_mean
get_u_cov
get_g_mean
get_ϕ_mean
get_u_mean_final
get_u_cov_final
get_g_mean_final
get_ϕ_mean_final
get_N_iterations
get_N_ens
get_accelerator
get_scheduler
get_process
get_rng
get_Δt
get_failure_handler
get_localizer
get_localizer_type
get_nan_tolerance
get_nan_row_values
list_update_groups_over_minibatch
```
## [Error metrics](@id errors_api)

```@docs
compute_average_rmse
compute_loss_at_mean
compute_average_unweighted_rmse
compute_unweighted_loss_at_mean
compute_bayes_loss_at_mean
compute_crps
compute_error!
get_error_metrics
get_error
lmul_obs_noise_cov
lmul_obs_noise_cov_inv
lmul_obs_noise_cov!
lmul_obs_noise_cov_inv!
```
## [Learning Rate Schedulers](@id scheduler_api)

```@docs
DefaultScheduler
MutableScheduler
EKSStableScheduler
DataMisfitController
calculate_timestep!
```
## Failure and NaN handling 

```@docs
FailureHandler
SampleSuccGauss
IgnoreFailures
sample_empirical_gaussian
split_indices_by_success
impute_over_nans
```

## [Process-specific](@id process_api)
```@docs
get_prior_mean
get_prior_cov
get_impose_prior
get_buffer
get_default_multiplicative_inflation
```