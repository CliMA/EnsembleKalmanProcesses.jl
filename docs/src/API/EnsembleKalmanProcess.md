# EnsembleKalmanProcess

```@meta
CurrentModule = EnsembleKalmanProcesses
```

```@docs
EnsembleKalmanProcess
FailureHandler
SampleSuccGauss
IgnoreFailures
get_u
get_g
get_ϕ
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
compute_error!
get_error
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
construct_initial_ensemble
update_ensemble!
sample_empirical_gaussian
split_indices_by_success
```

## [Learning Rate Schedulers](@id scheduler_api)

```@docs
DefaultScheduler
MutableScheduler
EKSStableScheduler
DataMisfitController
calculate_timestep!
```


## [Process-specific](@id process_api)
```@docs
get_prior_mean
get_prior_cov
get_impose_prior
get_buffer
get_default_multiplicative_inflation
```