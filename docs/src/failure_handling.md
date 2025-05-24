## [Handling forward model failures](@id failures)

In situations where the forward model ``\mathcal{G}`` represents a diagnostic of a complex computational model, there might be cases where for some parameter combinations ``\theta``, attempting to evaluate ``\mathcal{G}(\theta)`` may result in model failure. Some examples could be
1. A member is numerically unstable, or throws error exceptions
2. A member exceeds a user-defined wall-clock time for computation
3. A member's outputs produce (full or partial) values that are user-defined as "failure"
4. Possible corruption of data or compute nodes during HPC model evaluation

In such cases, this package offers the option for users to replace entries in the evaluation with `NaN` values, and then updates will continue with the following procedures
- Imputation: If only partial output of an ensemble member is `NaN`, then values may be redeemed with values derived from other ensemble members, or from user-defined values. Imputed members will be considered successful.
- Failure handling: The update equations are then modified to handle/replace/resample failed members. Crucially, this is done in a way that does not break parallelism (i.e., no re-running of failed members)

`EnsembleKalmanProcesses.jl` implements such modifications through the `FailureHandler` structure, an input to the `EnsembleKalmanProcess` constructor. Currently, the only failsafe modification available is `SampleSuccGauss()`, described in [Lopez-Gomez et al (2022)](https://doi.org/10.1029/2022MS003105).

!!! warning "These are last-resort handlers"
    These modifications are not a magic bullet. If large fractions of ensemble members fail during an iteration, this will degenerate the span of the ensemble, as one may not be able to replace lost information.

## Implementation

When available, the failure handling and imputation is active by default, see [here](@ref defaults) for the different options. Alternatively, one can use the `failure_handler_method` keyword

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions


# for e.g., EKI
J = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior
ekiobj = EnsembleKalmanProcess(
    initial_ensemble,
    y,
    obs_noise_cov,
    Inversion(),
    failure_handler_method = SampleSuccGauss())

# for e.g., UKI
ukiobj = EnsembleKalmanProcess(
    y
    obs_noise_cov,
    Unscented(prior),
    failure_handler_method = SampleSuccGauss())
```

## When has a particle "failed"?

### Full particle failure

The user determines if a ensemble member has failed and replace the output (column) of ``\mathcal{G}(\theta)`` with `NaN`s. The `FailureHandler` takes care of the rest.

### Partial particle failure (a.k.a Imputation)

If there is a partial failure within an output then the user sets only failed entries within a particle to `NaN`. The user can then use two keywords to determine the imputation approach, given to `EnsembleKalmanProcess`.

- `nan_tolerance`: If the number of `NaN`s in any observation is below a given `nan_tolerance` (default 10%) then the particle will still be considered successful (not handled by the `FailureHandler`) and the algorithm will impute `NaN`s using information from other ensemble members or user-defined values.
- `nan_rows_value`: if the same entry fails for all ensemble members, then imputing is not possible without additional information; here the user may supply a vector with the dimension of the output and this value will be imputed for all members in this row.

### Example of Imputation and Failure Handling
For example imagine we have a 7-dimensional output and 4 ensemble members. The latest run produces

```julia
g = 
7×4 Matrix{Float64}:
NaN           0.992373   NaN           -0.200025
-1.37133     0.0527899  NaN           -1.15243
NaN         NaN          NaN          NaN
NaN         NaN          NaN          NaN
-2.28799   NaN            0.614214     1.54217
0.582582    0.0744582   -1.37219      0.253971
0.620447    0.441834     0.0641471    0.490588
```
Now, given the user-defined settings in `EnsembleKalmanProcess(...)`
```julia
nan_tolerance = 0.5         # failure if > 50% NaN
nan_rows_value = collect(1:7) # replace failed row k with value k
```
The imputation will produce:
```julia
┌ Warning: In forward map ensemble g, detected 12 NaNs. 
│ Given nan_tolerance = 0.5 to determine failed members: 
│ - Ensemble members failed:       1 
│ - NaNs in successful members:    8 
│ - rows index set for imputation: [1, 3, 4, 5] 
│ - rows index entirely NaN:       [3, 4]
[ Info: Imputed 8 NaNs
7×4 Matrix{Float64}:
0.396174   0.992373   NaN          -0.200025     # `g[1, 1]`        -> `mean(g[1, [2, 4] ])`
-1.37133    0.0527899  NaN          -1.15243
3.0        3.0        NaN           3.0          # `g[3, [1, 2, 4]]`-> `3.0`
4.0        4.0        NaN           4.0          # `g[4, [1, 2, 4]]`-> `4.0`
-2.28799   -0.0438687    0.614214    1.54217     # `g[5, 2]`        -> `mean(g[5, [1, 3, 4])`
0.582582   0.0744582   -1.37219     0.253971
0.620447   0.441834     0.0641471   0.490588
```
Here,
- A warning is given to describe the context
- member 3 was deemed a failure(>50% NaN) and will be replaced with the failure handler, `SampleSuccGauss` during EKP update
- members 1, 2, and 4 were deemed successful, and so are imputed with values for the EKP update.
    
### [`SampleSuccGauss()` for EKI] (@id failure-eki)

The `SampleSuccGauss()` modification is based on updating all ensemble members with a distribution given by only the successful parameter ensemble. Let ``\Theta_{s,n}=[ \theta^{(1)}_{s,n},\dots,\theta^{(J_s)}_{s,n}]`` be the successful ensemble, for which each evaluation ``\mathcal{G}(\theta^{(j)}_{s,n})`` does not fail, and let ``\theta_{f,n}^{(k)}`` be the ensemble members for which the evaluation ``\mathcal{G}(\theta^{(k)}_{f,n})`` fails. The successful ensemble ``\Theta_{s,n}`` is updated to ``\Theta_{s,n+1}`` using expression (2), and each failed ensemble member as

```math
    \theta_{f,n+1}^{(k)} \sim \mathcal{N} \left({m}_{s, {n+1}}, \Sigma_{s, n+1} \right),
```

where

```math
    {m}_{s, {n+1}} = \dfrac{1}{J_s}\sum_{j=1}^{J_s} \theta_{s,n+1}^{(j)}, \qquad \Sigma_{s, n+1} = \mathrm{Cov}(\theta_{s, n+1}, \theta_{s, n+1}) + \kappa_*^{-1}\mu_{s,1}I_p.
```

Here, ``\kappa_*`` is a limiting condition number, ``\mu_{s,1}`` is the largest eigenvalue of the sample covariance ``\mathrm{Cov}(\theta_{s, n+1}, \theta_{s, n+1})`` and ``I_p`` is the identity matrix of size ``p\times p``.


### [`SampleSuccGauss()` for UKI] (@id failure-uki)

The `SampleSuccGauss()` modification is based on performing the UKI quadratures over the successful sigma points.
Consider the set of off-center sigma points ``\{\hat{\theta}\} = \{\hat{\theta}_s\} \cup \{\hat{\theta}_f\}`` where ``\hat{\theta}_{s}^{(j)}``,  ``j=1, \dots, J_s`` are successful members and ``\hat{\theta}_{f}^{(k)}`` are not. For ease of notation, consider an ordering of ``\{\hat{\theta}\}`` such that ``\{\hat{\theta}_s\}`` are its first ``J_s`` elements, and note that we deal with the central point ``\hat{\theta}^{(0)}`` separately. We estimate the covariances ``\mathrm{Cov}_q(\mathcal{G}_n, \mathcal{G}_n)`` and ``\mathrm{Cov}_q(\theta_{n}, \mathcal{G}_n)`` from the successful ensemble,

```math
   \tag{1} \mathrm{Cov}_q(\theta_n, \mathcal{G}_n) \approx \sum_{j=1}^{J_s}w_{s,j} (\hat{\theta}_{s, n}^{(j)} - \bar{\theta}_{s,n})(\mathcal{G}(\hat{\theta}_{s, n}^{(j)}) - \bar{\mathcal{G}}_{s,n})^T,
```

```math
   \tag{2} \mathrm{Cov}_q(\mathcal{G}_n, \mathcal{G}_n) \approx \sum_{j=1}^{J_s}w_{s,j} (\mathcal{G}(\hat{\theta}_{s, n}^{(j)}) - \bar{\mathcal{G}}_{s,n})(\mathcal{G}(\hat{\theta}_{s, n}^{(j)}) - \bar{\mathcal{G}}_{s,n})^T,
```

where the weights at each successful sigma point are scaled up, to preserve the sum of weights,
```math
    w_{s,j} = \left(\dfrac{\sum_{i=1}^{2p} w_i}{\sum_{k=1}^{J_s} w_k}\right)w_j.
```

In equations (1) and (2), the means ``\bar{\theta}_{s,n}`` and ``\bar{\mathcal{G}}_{s,n}`` must be modified from the original formulation if the central point ``\hat{\theta}^{(0)}=m_n`` results in model failure. If this is the case, then an average is taken across the other (successful) ensemble members

```math
   \bar{\theta}_{s,n} =
\dfrac{1}{J_s}\sum_{j=1}^{J_s}\hat{\theta}_{s, n}^{(j)}, \qquad   \bar{\mathcal{G}}_{s,n} =
\dfrac{1}{J_s}\sum_{j=1}^{J_s}\mathcal{G}(\hat{\theta}_{s, n}^{(j)}).
```
