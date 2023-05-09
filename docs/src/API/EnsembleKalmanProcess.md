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
construct_initial_ensemble
update_ensemble!
sample_empirical_gaussian
split_indices_by_success
```

## Learning Rate Schedulers

```@docs
DefaultScheduler
MutableScheduler
EKSStableScheduler
DataMisfitController
calculate_timestep!
```


