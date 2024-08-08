## Default Configurations

Sensible defaults have been chosen for the methodology to give broadly the best solution. The configurations can be revealed by adding the keyword
```julia
EnsembleKalmanProcess(...,verbose = true)
```
To obtain the defaults, one constructs a type of Ensemble Kalman Process with
```julia
EnsembleKalmanProcess(initial_parameters, observation, process)
```
and the following configurations (listed below) will be automatically created depending on the `process` type chosen. Please see the relevant documentation pages for each configurable.

!!! info "Recommended"
    For the simplest and most flexible update we recommend the `Inversion()` process. 
    ```julia
    EnsembleKalmanProcess(initial_parameters, observation, Inversion())
    ```

##  `process <: Inversion` 

```julia
Dict(
    "scheduler" => DataMisfitController(terminate_at = 1),
    "localization_method" => SECNice(),
    "failure_handler_method" => SampleSuccGauss(),
    "accelerator" => NesterovAccelerator(),
)
```

## `process <: TransformInversion`

```julia
Dict(
    "scheduler" => DataMisfitController(terminate_at = 1),
    "localization_method" => NoLocalization(),
    "failure_handler_method" => SampleSuccGauss(),
    "accelerator" => DefaultAccelerator(),
)
```

## `process <: SparseInversion`

```julia
Dict(
    "scheduler" => DefaultScheduler(),
    "localization_method" => SECNice(),
    "failure_handler_method" => SampleSuccGauss(),
    "accelerator" => DefaultAccelerator(),
)
```

## `process <: Sampler`

```julia
Dict(
    "scheduler" => EKSStableScheduler(1.0, eps()),
    "localization_method" => NoLocalization(),
    "failure_handler_method" => IgnoreFailures(),
    "accelerator" => DefaultAccelerator(),
)
```

## `process <: Unscented`

```julia
Dict(
    "scheduler" => DataMisfitController(terminate_at = 1),
    "localization_method" => NoLocalization(),
    "failure_handler_method" => SampleSuccGauss(),
    "accelerator" => DefaultAccelerator(),
)
```

## "Vanilla" settings - turning off features
All the configurable features can be set with keywords. The following object will be created with no additional features.

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Localizers
EnsembleKalmanProcess(
    initial_parameters,
    observation,
    process,
    scheduler = DefaultScheduler(1), # constant timestep size 1
    localization_method = Localizers.NoLocalization(), # no localization
    failure_handler_method = IgnoreFailures(), # no failure handling
    accelerator = DefaultAccelerator(), # no acceleration
)
