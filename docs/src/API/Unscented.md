# Unscented Kalman Inversion

```@meta
CurrentModule = EnsembleKalmanProcesses
```

```@docs
Unscented
construct_sigma_ensemble
construct_mean
construct_cov
update_ensemble_prediction!
update_ensemble_analysis!
construct_initial_ensemble(prior::ParameterDistribution, process::UorTU) where {UorTU <: Union{Unscented, TransformUnscented}}

```