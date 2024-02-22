# Troubleshooting and Workflow Tips

## High Failure Rate

While some EKI variants include failure handlers, excessively high failure rates (i.e., > 80%) can lead to inversions finding local minima or failing to converge. To address this:

- **Stabilize the Forward Model**: Ensure the forward model remains stable for small parameter perturbations in offline tests.
- **Adjust Priors**: Reduce the uncertainty in priors. Priors with large variances can lead to forward evaluations that deviate significantly from the known prior means, increasing the likelihood of failures.
- **Increase Ensemble Size**: Without localization (or other methods that break the subspace property), the ensemble size should generally exceed the number of parameters being optimized. The ensemble size needs to be large enough to ensure a sufficient number of successful runs, given the failure rate.
- **Increase Preconditioner Retry Count**: If using a preconditioner that recursively draws from the prior until a stable run is achieved, increase the retry count (typically to more than 5).
- **Implement Parameter Inflation**: Prevent the ensemble from collapsing prematurely by adding parameter inflation.

## Loss Doesn't Converge

If the loss decreases too slowly or diverges:

- **Check for Failures**: Refer to the suggestions for handling a high failure rate.
- **Adjust the Artificial Timestep**: For indirect learning problems involving neural networks, larger timesteps [O(10)] are generally more effective and using variable timesteppers (e.g., DMC) tends to yield the best results.
- **If Batching, Increase Batch Size**: If the loss is too noisy and convergence is slow, consider increasing the batch size.
- **Check Observation Noise in Data Space**: Ensure that noise estimates are realistic and consistent across variables with different dimensions and variability characteristics. Observation noise that is unrealistically large for a given variable or data point may prevent convergence to solutions that closely fit the data. Carefully base noise estimates on empirical data or domain knowledge, and try reducing noise if the previous suggestions don’t work.
- **Reevaluate Loss Function**: Consider exploring alternative loss functions with different variables.
