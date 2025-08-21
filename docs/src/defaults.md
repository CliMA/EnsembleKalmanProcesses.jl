## [Default Configurations](@id defaults)

## Recommended Ensemble Size

The ensemble size is generally not determinable in advance.
However there are several rules of thumb for calibrating a parameter vector ``\theta`` that can be used as a starting point.

Parameter dimension        | Ensemble size 
--------------------------------|----------------------------------------------------------------
``\mathrm{dim}(\theta)\leq 10`` | ``N_{\mathrm{ens}} \geq 10 \cdot \mathrm{dim}(\theta)``
``10 \leq \mathrm{dim}(\theta)\leq 100`` | ``N_{\mathrm{ens}} = 100``
``100\leq \mathrm{dim}(\theta)``| ``N_{\mathrm{ens}} = 100`` and [SEC](@ref localization)

!!! note "for the `Unscented` process"
    [UKI](@ref uki) and [UTKI](@ref utki) always create an ensemble size proportional to `` \mathrm{dim}(\theta)``. It is not configurable by the user, and is retrievable from an `EnsembleKalmanProcess` object `ekp` using `get_N_ens(ekp)`.

### Quick links!
- What does `scheduler = ...` do? See [here.](@ref learning-rate-schedulers)
- What does `localization_method = ...` do? See [here.](@ref localization) and our [example](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples/Localization)
- What does `failure_handler_method = ...` do? See [here](@ref failures)
- What does `accelerator = ...` do? See [here](@ref accelerators), and our [examples](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples/Accelerators)

## Prebuilt defaults

Defaults have been chosen for the methodology based on prior experience. The configurations can be revealed by adding the keyword
```julia
EnsembleKalmanProcess(..., verbose = true)
```
To use the defaults, one constructs an Ensemble Kalman Process with
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

## `process <: GaussNewtonInversion` 
Process documentation [here](@ref gnki)
```julia
scheduler = DataMisfitController(terminate_at = 1)
localization_method = Localizers.SECNice()
failure_handler_method = SampleSuccGauss()
accelerator = NesterovAccelerator()
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

## `process <: TransformUnscented` 
Process documentation [here](@ref utki)

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