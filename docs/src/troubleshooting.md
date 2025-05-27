# [Troubleshooting and Workflow Tips](@id troubleshooting)

## [Getting the results](@id get-results)

Data is stored within the `EnsembleKalmanProcess`. Accessing results is done via our API of `getter` functions, and transformations.

The most common of these is
```julia
# given an
# EnsembleKalmanProcess `ekp`
# prior                 `prior`

# getting the "latest" parameter ensemble, or at a chosen iteration
ϕ = get_ϕ_final(prior, ekp)
ϕ = get_ϕ(prior, ekp, iteration)

# getting the "latest" output ensemble, or at a chosen iteration
g = get_g_final(ekp)
g = get_g(ekp, iteration)

# getting the "latest" data vector, or at a chosen iteration [if e.g. minibatching]
y = get_obs(ekp)
y = get_obs(ekp, iteration)

# getting the "latest" observational noise covariance, or at a chosen iteration [if e.g. minibatching]
Γ = get_obs_noise_cov(ekp) 
Γ = get_obs_noise_cov(ekp, build=false) # do not build the matrix, keep it blocked 
Γ = get_obs_noise_cov(ekp, iteration) 

# get the computed error metrics over iterations from `compute_error!`
metrics = get_error_metrics(ekp)

# get the corresponding algorithm time for the iterations performed
Δt = get_algorithm_time(ekp) 


# get the latest computational parameters
u = get_u_final(ekp)  
# where ϕ = transform_unconstrained_to_unconstrained.(prior, get_u_final(ekp))
```


## Convergence diagnosis and plotting

Information about convenient plotting tools, or error metrics computed during updates, please see [here.](@ref visualization)

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
- **If Batching, Increase Batch Size**: Users employing [minibatching](@ref observations) (using subsamples of the full dataset in each EKI iteration) should consider modifying the batch size. 
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
  Warning: Termination condition of scheduler `DataMisfitController` has been exceeded, returning `true` from `update_ensemble!` and preventing futher updates. Set on_terminate="continue" in `DataMisfitController` to ignore termination
  ```
The `DataMisfitController` is an adaptive scheduler that can terminate the algorithm at a given value of algorithm time (rather than juat a given number of iterations). See [here](@ref learning-rate-schedulers) for details on changing the termination condition. Or how to handle this in your iteration loop.

- ```julia
  Warning: Acceleration is experimental for Sampler processes and may affect convergence.
  ```
This is found when providing something other than `accelerator = DefaultAccelerator()` in EKS.
- ```julia
  Info: 1 particle failure(s) detected. Handler used: IgnoreFailures.
  Info: 1 particle failure(s) detected. Handler used: SampleSuccGauss.
  ```
Both these messages arise when the EKP `update_ensemble!` has detected `NaN`s in the forward map evaluation array. One can choose how to handle failed ensemble members with the EKP keyword `failure_handler_method` for more information on the failure handling methods see [here](@ref failures). Note that EKS does not yet have a consistent failure handler method.

- ```julia
  Warning: Detected 2 clashes where forward map evaluations are exactly equal (and not NaN), this is likely to cause `LinearAlgebra` difficulty. Please check forward evaluations for bugs.
  ```
This message arises when forward map evaluations from different paramters produce identical outputs and is usually due to (1) output-clipping practices, or (2) user-error in producing the output matrix for the ensemble and should be checked, as it may cause errors. (If not, the implication is that the model output is completely independent of the parameters) 