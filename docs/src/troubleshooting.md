# [Troubleshooting and Workflow Tips](@id troubleshooting)

## High failure rate

While some EKI variants include failure handlers, excessively high failure rates (i.e., > 80%) can lead to inversions finding local minima or failing to converge. To address this:

- **Stabilize the Forward Model**: Ensure the forward model remains stable for small parameter perturbations in offline tests. If the forward model is unstable for most of the parameter space, it is challenging to explore it with a calibration method.
- **Adjust Priors**: Reduce the uncertainty in priors. Priors with large variances can lead to forward evaluations that deviate significantly from the known prior means, increasing the likelihood of failures.
- **Increase Ensemble Size**: Without [localization](@ref localization) or other methods that break the subspace property, the ensemble size should generally exceed the number of parameters being optimized. The ensemble size needs to be large enough to ensure a sufficient number of successful runs, given the failure rate.
- **Consider a Preconditioner**: While not currently a native feature in EKP, consider using a preconditioning method to find stable parameter pairs before the first iteration. A preconditioner, applied in each ensemble member, recursively draws from the prior distribution until a stable parameter pair is achieved. The successful parameter pairs serve as the parameter values for the first iteration. Depending on the stability of the forward model,this may need to be as high as 5-10 retries.
- **Implement Parameter Inflation**: High failure rates in the initial iterations can lead to rapid collapse of ensemble members. Prevent the ensemble from collapsing prematurely by adding parameter inflation. For more, see [inflation](@ref inflation)

## Loss does not converge or final fits are inadequate

If either the loss decreases too slowly/diverges or the final fits appear inadequate:

- **Check Observation Noise in Data Space**: Ensure that noise estimates are realistic and consistent across variables with different dimensions and variability characteristics. Observation noise that is unrealistically large for a given variable or data point may prevent convergence to solutions that closely fit the data. Carefully base noise estimates on empirical data or domain knowledge, and try reducing noise if the previous suggestions don’t work. This is especially common if using ``\sigma^2 * I`` as artificial noise. Even if ``u`` appears incorrect, it is advisable to examine the graphs of ``G(u)`` compared to  ``y \pm 2\sigma`` to determine if the forward map lies within the noise level. If it does, further convergence cannot be achieved without reducing the noise or altering the loss function.
- **Check for Failures**: Refer to the suggestions for handling a high failure rate.
- **Adjust the Artificial Timestep**: For indirect learning problems involving neural networks, larger timesteps [O(10)] are generally more effective and using variable timesteppers (e.g., `DataMisfitController()`) tends to yield the best results. For scheduler options, see [scheduler](@ref learning-rate-schedulers) docs.
- **If Batching, Increase Batch Size**: While not natively implemented within the EKP package, users employing mini-batching (using subsamples of the full dataset in each EKI iteration) should consider modifying the batch size. If the loss is too noisy and convergence is slow, consider increasing the batch size. See [inflation](@ref inflation) if using mini-batching with inflation. 
- **Reevaluate Loss Function**: Consider exploring alternative loss functions with different variables.
- **Structural Model Errors**: If these troubleshooting tips do not work, remaining discrepancies might suggest inherent structural errors between the model and the data, which could lead to trade-offs in parameter estimation. Modifications may be needed to the underlying forward model. 

## I have a model ``\Psi``. But how do I design my forward map G?
- Ensure prior means are chosen appropriately, and that any hard [constraints](@ref parameter-distributions) (i.e., parameter values must be positive) are enforced.
- Start with a perfect model experiment, where an attempt is made to recover known parameter values in ``\Psi`` through calibration, to learn about what outputs from ``\Psi`` are sensitive to the parameters.
- For time-evolving systems, consider aggregating data through spatial or temporal averaging, rather than using the full timeseries. 
- Find out which observational data are available for the problem at hand, and what observational noise is provided for measuring instruments.

## [Common warning/error messages](@id common-messages)
- ```julia
  Info: "Termination condition of scheduler `DataMisfitController` will be exceeded during the next iteration."
  Warning: Termination condition of timestepping scheme `DataMisfitController` has been exceeded, returning `true` from `update_ensemble!` and preventing futher updates. Set on_terminate="continue" in `DataMisfitController` to ignore termination
```
The `DataMisfitController` is an adaptive scheduler/timestepper that terminates at a given value of algorithm time (rather than always a maximum iteration). See [here](@ref learning-rate-schedulers) for details on changing the termination condition. Or how to handle this in your iteration loop

- ```julia
Warning: Acceleration is experimental for Sampler processes and may affect convergence.
```
This is found when providing something other than `accelerator = DefaultAccelerator()` in EKP
- ```julia
Info: 1 particle failure(s) detected. Handler used: IgnoreFailures.
Info: 1 particle failure(s) detected. Handler used: SampleSuccGauss.
```
Both these messages have detected `NaN`s in the forward map evaluation array. One can choose how to handle failed ensemble members with the EKP keyword `failure_handler_method` for more information on the failure handling methods see [here for EKI/ETKI/SEKI](@ref failure-eki) or [here for UKI](@ref failure-uki). EKS does not yet have a consistent failure handler method.