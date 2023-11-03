This page documents ensemble Kalman inversion (EKI), as well as two variants, [ensemble transform Kalman inversion](@ref etki) (ETKI) and [sparsity-inducing ensemble Kalman inversion](@ref seki) (SEKI).

# Ensemble Kalman Inversion

One of the ensemble Kalman processes implemented in `EnsembleKalmanProcesses.jl` is ensemble
Kalman inversion ([Iglesias et al, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)).
Ensemble Kalman inversion (EKI) is a derivative-free ensemble optimization method that seeks
to find the optimal parameters ``\theta \in \mathbb{R}^p`` in the inverse problem defined by the data-model relation

```math
\tag{1} y = \mathcal{G}(\theta) + \eta ,
```

where ``\mathcal{G}`` denotes the forward map, ``y \in \mathbb{R}^d`` is the vector of observations
and ``\eta  \in \mathbb{R}^d`` is additive noise. Note that ``p`` is the
size of the parameter vector ``\theta`` and ``d`` the size of the observation vector ``y``. Here, we take ``\eta \sim \mathcal{N}(0, \Gamma_y)`` from a ``d``-dimensional Gaussian with zero mean and covariance matrix ``\Gamma_y``.  This noise structure aims to represent the correlations between observations.

The optimal parameters ``\theta^* \in \mathbb{R}^p`` given relation (1) minimize the loss

 ```math
\mathcal{L}(\theta, y) = \langle \mathcal{G}(\theta) - y \, , \, \Gamma_y^{-1} \left ( \mathcal{G}(\theta) - y \right ) \rangle,
```

which can be interpreted as the negative log-likelihood given a Gaussian likelihood.

Denoting the parameter vector of the ``j``-th ensemble member at the ``n``-th iteration as ``\theta^{(j)}_n``, its update equation from ``n`` to ``n+1`` under EKI is

```math
\tag{2} \theta_{n+1}^{(j)} = \theta_{n}^{(j)} - \dfrac{\Delta t_n}{J}\sum_{k=1}^J \left \langle \mathcal{G}(\theta_n^{(k)}) - \bar{\mathcal{G}}_n \, , \, \Gamma_y^{-1} \left ( \mathcal{G}(\theta_n^{(j)}) - y \right ) \right \rangle \theta_{n}^{(k)} ,
```

where the subscript ``n=1, \dots, N_{\rm it}`` indicates the iteration, ``J`` is the number of
members in the ensemble, ``\bar{\mathcal{G}}_n`` is the mean value of ``\mathcal{G}(\theta_n)``
across ensemble members,

```math
\bar{\mathcal{G}}_n = \dfrac{1}{J}\sum_{k=1}^J\mathcal{G}(\theta_n^{(k)}) ,
```

and angle brackets denote the Euclidean inner product. By multiplying with ``\Gamma_y^{-1}``
we render the inner product non-dimensional.

The EKI algorithm is considered converged when the ensemble achieves sufficient consensus/collapse
in parameter space. The final estimate ``\bar{\theta}_{N_{\rm it}}`` is taken to be the ensemble
mean at the final iteration,

```math
\bar{\theta}_{N_{\rm it}} = \dfrac{1}{J}\sum_{k=1}^J\theta_{N_{\rm it}}^{(k)}.
```

For typical applications, a near-optimal solution ``\theta`` can be found after as few as 10 iterations of the algorithm, or ``10\cdot J`` evaluations of the forward model ``\mathcal{G}``. The basic algorithm requires ``J \geq p``, and better performance is often seen with larger ensembles; a good rule of thumb is to start with ``J=10p``. The algorithm also extends to ``J < p`` , using localizers to maintain performance in these situations (see the Localizers.jl module).

## Constructing the Forward Map

The forward map ``\mathcal{G}`` maps the space of unconstrained parameters ``\theta \in \mathbb{R}^p`` to the space of outputs ``y \in \mathbb{R}^d``. In practice, the user may not have access to such a map directly. Consider a situation where the goal is to learn a set of parameters ``\phi`` of a dynamical model ``\Psi: \mathbb{R}^p \rightarrow \mathbb{R}^o``, given observations ``y \in \mathbb{R}^d`` and a set of constraints on the value of ``\phi``. Then, the forward map may be constructed as

```math
\mathcal{G} = \mathcal{H} \circ \Psi \circ \mathcal{T}^{-1},
```

where ``\mathcal{H}: \mathbb{R}^o \rightarrow \mathbb{R}^d`` is the observation map and ``\mathcal{T}`` is the transformation map from constrained to unconstrained parameter spaces, such that ``\mathcal{T}(\phi) = \theta``. A family of standard transformation maps and their inverse are available in the `ParameterDistributions` module.

## Creating the EKI Object

An ensemble Kalman inversion object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Inversion()` process type.

Creating an ensemble Kalman inversion object requires as arguments:
 1. An initial parameter ensemble, `Array{Float, 2}` of size `[p × J]`;
 2. The mean value of the observed outputs, a vector of size `[d]`;
 3. The covariance of the observational noise, a matrix of size `[d × d]`;
 4. The `Inversion()` process type.

A typical initialization of the `Inversion()` process takes a user-defined `prior`, a summary of the observation statistics given by the mean `y` and covariance `obs_noise_cov`, and a desired number of members in the ensemble,
```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

J = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior

ekiobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Inversion())
```

See the [Prior distributions](@ref parameter-distributions) section to learn about the construction of priors in EnsembleKalmanProcesses.jl. The prior is assumed to be over the unconstrained parameter space where ``\theta`` is defined. For applications where enforcing parameter bounds is necessary, the `ParameterDistributions` module provides functions to map from constrained to unconstrained space and vice versa. 

## Updating the Ensemble

Once the ensemble Kalman inversion object `ekiobj` has been initialized, any number of updates can be performed using the inversion algorithm.

A call to the inversion algorithm can be performed with the `update_ensemble!` function. This function takes as arguments the `ekiobj` and the evaluations of the forward map at each member of the current ensemble. The `update_ensemble!` function then stores the new updated ensemble and the inputted forward map evaluations in `ekiobj`. 

A typical use of the `update_ensemble!` function given the ensemble Kalman inversion object `ekiobj`, the dynamical model `Ψ` and the observation map `H` is
```julia
# Given:
# Ψ (some black box simulator)
# H (some observation of the simulator output)
# prior (prior distribution and parameter constraints)

N_iter = 20 # Number of steps of the algorithm

for n in 1:N_iter
    ϕ_n = get_ϕ_final(prior, ekiobj) # Get current ensemble in constrained "ϕ"-space
    G_n = [H(Ψ(ϕ_n[:, i])) for i in 1:J]
    g_ens = hcat(G_n...) # Evaluate forward map 
    update_ensemble!(ekiobj, g_ens) # Update ensemble
end
```

In the previous update, note that the parameters stored in `ekiobj` are given in the unconstrained
Gaussian space where the EKI algorithm is performed. The map ``\mathcal{T}^{-1}`` between this unconstrained
space and the (possibly constrained) physical space of parameters is encoded in the `prior` object. The
dynamical model `Ψ` accepts as inputs the parameters in (possibly constrained) physical space, so it is
necessary to use the getter `get_ϕ_final` which applies `transform_unconstrained_to_constrained` to the ensemble. See the
[Prior distributions](@ref parameter-distributions) section for more details on parameter transformations.   

## Solution

The EKI algorithm drives the initial ensemble, sampled from the prior, towards the support region of the posterior distribution. The algorithm also drives the ensemble members towards consensus. The optimal parameter `θ_optim` found by the algorithm is given by the mean of the last ensemble (i.e., the ensemble after the last iteration),

```julia
θ_optim = get_u_mean_final(ekiobj) # optimal parameter
```
To obtain the optimal value in the constrained space, we use the getter with the constrained prior as input
```julia
ϕ_optim = get_ϕ_mean_final(prior, ekiobj) # the optimal physical parameter value
```

## Handling forward model failures

In situations where the forward model ``\mathcal{G}`` represents a diagnostic of a complex computational model, there might be cases where for some parameter combinations ``\theta``, attempting to evaluate ``\mathcal{G}(\theta)`` may result in model failure (defined as returning a `NaN` from the point of view of this package). In such cases, the EKI update equation (2) must be modified to handle model failures.

`EnsembleKalmanProcesses.jl` implements such modifications through the `FailureHandler` structure, an input to the `EnsembleKalmanProcess` constructor. Currently, the only failsafe modification available is `SampleSuccGauss()`, described in [Lopez-Gomez et al (2022)](https://doi.org/10.1029/2022MS003105).

To employ this modification, construct the EKI object as

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

J = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior

ekiobj = EnsembleKalmanProcess(
    initial_ensemble,
    y,
    obs_noise_cov,
    Inversion(),
    failure_handler_method = SampleSuccGauss())
```

!!! info "Forward model requirements when using FailureHandlers"
    The user must determine if a model run has "failed", and replace the output ``\mathcal{G}(\theta)`` with `NaN`. The `FailureHandler` takes care of the rest.

A description of the algorithmic modification is included below.

### `SampleSuccGauss()`

The `SampleSuccGauss()` modification is based on updating all ensemble members with a distribution given by only the successful parameter ensemble. Let ``\Theta_{s,n}=[ \theta^{(1)}_{s,n},\dots,\theta^{(J_s)}_{s,n}]`` be the successful ensemble, for which each evaluation ``\mathcal{G}(\theta^{(j)}_{s,n})`` does not fail, and let ``\theta_{f,n}^{(k)}`` be the ensemble members for which the evaluation ``\mathcal{G}(\theta^{(k)}_{f,n})`` fails. The successful ensemble ``\Theta_{s,n}`` is updated to ``\Theta_{s,n+1}`` using expression (2), and each failed ensemble member as

```math
    \theta_{f,n+1}^{(k)} \sim \mathcal{N} \left({m}_{s, {n+1}}, \Sigma_{s, n+1} \right),
```

where

```math
    {m}_{s, {n+1}} = \dfrac{1}{J_s}\sum_{j=1}^{J_s} \theta_{s,n+1}^{(j)}, \qquad \Sigma_{s, n+1} = \mathrm{Cov}(\theta_{s, n+1}, \theta_{s, n+1}) + \kappa_*^{-1}\mu_{s,1}I_p.
```

Here, ``\kappa_*`` is a limiting condition number, ``\mu_{s,1}`` is the largest eigenvalue of the sample covariance ``\mathrm{Cov}(\theta_{s, n+1}, \theta_{s, n+1})`` and ``I_p`` is the identity matrix of size ``p\times p``.

!!! warning
    This modification is not a magic bullet. If large fractions of ensemble members fail during an iteration, this will degenerate the span of the ensemble.

# [Ensemble Transform Kalman Inversion](@id etki)

Ensemble transform Kalman inversion (ETKI) is a variant of EKI based on the ensemble transform Kalman filter ([Bishop et al., 2001](http://doi.org/10.1175/1520-0493(2001)129<0420:ASWTET>2.0.CO;2)). It is a form of ensemble square-root inversion, and was previously implemented in [Huang et al., 2022](http://doi.org/10.1088/1361-6420/ac99fa). The main advantage of ETKI over EKI is that it has better scalability as the observation dimension grows: while the naive implementation of EKI scales as ``\mathcal{O}(p^3)`` in the observation dimension ``p``, ETKI scales as ``\mathcal{O}(p)``. This, however, refers to the online cost. ETKI may have an offline cost of ``\mathcal{O}(p^3)`` if ``\Gamma`` is not easily invertible; see below.

The major disadvantage of ETKI is that it cannot be used with localization or sampling error correction. ETKI also requires the inverse observation noise covariance, ``\Gamma^{-1}``. In typical applications, when ``\Gamma`` is diagonal, this will be cheap to compute; however, if ``p`` is very large and ``\Gamma`` has non-trivial cross-covariance structure, computing the inverse may be prohibitively expensive.

## Using ETKI

An ETKI struct can be created using the `EnsembleKalmanProcess` constructor by specifying the `TransformInversion` process type, with ``\Gamma^{-1}`` passed as an argument: 

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

J = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior

ekiobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov,
                               TransformInversion(inv(obs_noise_cov)))
```

The rest of the inversion process is the same as for regular EKI.

# [Sparsity-Inducing Ensemble Kalman Inversion](@id seki)

We include Sparsity-inducing Ensemble Kalman Inversion (SEKI) to add approximate ``L^0`` and ``L^1`` penalization to the EKI ([Schneider, Stuart, Wu, 2020](https://doi.org/10.48550/arXiv.2007.06175)).

!!! warning
    The algorithm suffers from robustness issues, and therefore we urge caution in using the tool

