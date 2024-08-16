## [Default Configurations](@id defaults)

## Recommended Ensemble Size

The ensemble size is generally not determinable in advance.
However there are several rules of thumb, for calibrating a parameter vector `\theta`, that can be used as a starting point.

``\mathrm{dim}(\theta)``        | ``N_{\mathrm{ens}}`` 
--------------------------------|----------------------------------------------------------------
``\mathrm{dim}(\theta)\leq 10`` | ``N_{\mathrm{ens}} = 10\mathrm{dim}(\theta)``
``\mathrm{dim}(\theta)\geq 10`` | ``N_{\mathrm{ens}} = 100``
``\mathrm{dim}(\theta)\geq 100``| ``N_{\mathrm{ens}} = 100`` and use [sampling error correction](@id localization)

!!! note "for the `Unscented` process"
    The [UKI method](@ref uki) always creates an ensemble size proportional to the input dimension, and it is not configurable by the user.

## Prebuilt defaults

Sensible defaults have been chosen for the methodology to give broadly the best solution. The configurations can be revealed by adding the keyword
```julia
EnsembleKalmanProcess(...,verbose = true)
```
To use the defaults, one constructs a type of Ensemble Kalman Process with
```julia
EnsembleKalmanProcess(initial_parameters, observation, process)
```
and the following configurations (listed below) will be automatically created depending on the `process` type chosen, they are listed as keyword arguments that will be automatically added into `EnsembleKalmanProcess()` on creation. 

!!! info "Recommended process"
    For the simplest and most flexible update we recommend the `Inversion()` process. 
    ```julia
    EnsembleKalmanProcess(initial_parameters, observation, Inversion())
    ```

Please see the relevant documentation pages for each configurable if you wish to modify them. 

## `process <: Inversion` 
Process documentation [here](@ref eki)
```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.SECNice()
failure_handler_method = SampleSuccGauss()
accelerator = NesterovAccelerator()
```

## `process <: TransformInversion`
Process documentation [here](@ref etki)
```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.NoLocalization()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## `process <: SparseInversion`
Process documentation [here](@ref seki)

```julia
scheduler = DefaultScheduler()
localization_method = Localizers.SECNice()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## `process <: Sampler`
Process documentation [here](@ref eks)
```julia
scheduler = EKSStableScheduler(1.0, eps())
localization_method = Localizers.NoLocalization()
failure_handler_method = IgnoreFailures()
accelerator = DefaultAccelerator()
```

## `process <: Unscented`
Process documentation [here](@ref uki)

```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.NoLocalization()
failure_handler_method = SampleSuccGauss()
accelerator = DefaultAccelerator()
```

## "Vanilla" settings: How to turn off features

As the defaults now implement recent features. The following snippet shows how one can use keyword arguments to construct an EKP with no additional features or variants.
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

!!! note
    You will need to call the `Localizers` module via
    ```julia
    using EnsembleKalmanProcesses.Localizers
    ```
    to get the `localization_method` structures