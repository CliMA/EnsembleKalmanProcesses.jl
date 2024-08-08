## Default Configurations

Sensible defaults have been chosen for the methodology to give broadly the best solution. The configurations can be revealed by adding the keyword
```julia
EnsembleKalmanProcess(...,verbose = true)
```
To use the defaults, one constructs a type of Ensemble Kalman Process with
```julia
EnsembleKalmanProcess(initial_parameters, observation, process)
```
and the following configurations (listed below) will be automatically created depending on the `process` type chosen, they are listed as keyword arguments that will be automatically added into `EnsembleKalmanProcess()` on creation. 

!!! info "Recommended"
    For the simplest and most flexible update we recommend the `Inversion()` process. 
    ```julia
    EnsembleKalmanProcess(initial_parameters, observation, Inversion())
    ```
Please see the relevant documentation pages for each configurable if you wish to modify them.
##  `process <: Inversion` 

```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.SECNice()
failure_handler_method = SampleSuccGauss()
accelerator = NesterovAccelerator()
```

## `process <: TransformInversion`

```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.NoLocalization()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## `process <: SparseInversion`

```julia
scheduler = DefaultScheduler()
localization_method = Localizers.SECNice()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## `process <: Sampler`

```julia
scheduler = EKSStableScheduler(1.0, eps())
localization_method = Localizers.NoLocalization()
failure_handler_method = IgnoreFailures()
accelerator = DefaultAccelerator()
```

## `process <: Unscented`

```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.NoLocalization()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## "Vanilla" settings - how to turn off features
The following object will be created with no additional features.

To modify the defaults the following modules should be loaded in:
```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Localizers
```

```
EnsembleKalmanProcess(
    initial_parameters,
    observation,
    process,
    scheduler = DefaultScheduler(1), # constant timestep size 1
    localization_method = Localizers.NoLocalization(), # no localization
    failure_handler_method = IgnoreFailures(), # no failure handling
    accelerator = DefaultAccelerator(), # no acceleration
)
```