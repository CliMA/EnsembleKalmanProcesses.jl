This page documents ensemble Kalman inversion (EKI), as well as two variants, [ensemble transform Kalman inversion](@ref etki) (ETKI) and [sparsity-inducing ensemble Kalman inversion](@ref seki) (SEKI).

# [Ensemble Kalman Inversion](@id eki)

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
# [`Inversion()` vs `Inversion(prior)`](@id finite-vs-infinite-time)

!!! note "Finite-time vs infinite-time"
    Deeper description of these algorithms is discussed in detail in, for example, Section 4.5 of [Calvello, Reich, Stuart](https://arxiv.org/pdf/2209.11371)). Finite-time algorithms have also been called "transport" algorithms, and infinite-time algorithms are also known as prior-enforcing, or Tikhonov EKI [Chada, Stuart, Tong](https://doi.org/10.1137/19M1242331).

Thus far, we have presented the finite-time algorithm `Inversion()`. The infinite-time variant `Inversion(prior)` algorithm has two key distinctions.
1. The initial distribution does not need to come from the prior. 
2. The particle distribution mean converges to the maximum a-posteriori estimator as ``T\to \infty`` (not via an [early-termination condition](@ref early-terminate))
It is implemented as follows (here, for three parameters)
```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
# given `y` `obs_noise_cov` and `prior`

J = 50  # number of ensemble members
initial_dist = constrained_gaussian("not-the-prior", 0, 1, -Inf, Inf, repeats=3)
initial_ensemble = construct_initial_ensemble(inital_dist, J) # Initialize ensemble from prior

ekiobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Inversion(prior))
```

One can see this in-action with the finite- vs infinite-time comparison example [here](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/LossMinimization/), which was used to produce the plots below:

**Left: `Inversion` (finite-time), Right: `Inversion(prior)` (infinite-time, initialized off-prior)**
```@raw html
<img src="assets/animations/animated_inversion-finite.gif" width="300"> <img src="assets/animations/animated_inversion-infinite.gif" width="300"> 
```
Comparative behaviour. 
1. **Initialization:** `Inversion()` must be initialized from the prior, `Inversion(prior)` can still find the posterior when initialized off-prior. This might be useful when the prior is very broad and can enter, for example, regions of instability of the users forward model
2. **Prior information:** `Inversion()` only contains prior information due to its initialization, `Inversion(prior)` enforces the prior at every iteration.
3. **Solution**: `Inversion()` terminated at ``T=1`` (implemented by default) obtains an accurate MAP estimate, the ensemble spread at exactly ``T=1`` can represent a snapshot of the true (Gaussian-approximated) uncertainty. `Inversion(prior)` obtains this in the limit ``T\to\infty``, and undergoes collapse providing no uncertainty information.
4. **Trust in prior** `Inversion()`, when iterated beyond ``T=1`` will lose prior information and thus move to find the MLE (minimize the data-misfit only) at ``T\to\infty``, this behaviour might be useful if the prior information is overly confident.  
5. **Efficiency**: `Inversion()` is more efficient that `Inversion(prior)` as enforcing the prior in the infinite-time algorithm is performed via extending the linear systems to be solved. Performance is also impacted (positively or negatively) by the choice of initial distribution in the `Inversion(prior)`

One can learn more about the early termination for finite-time algorithms [here](@ref early-terminate).

# [Output-scalable variant: Ensemble Transform Kalman Inversion](@id etki)

Ensemble transform Kalman inversion (ETKI) is a variant of EKI based on the ensemble transform Kalman filter ([Bishop et al., 2001](http://doi.org/10.1175/1520-0493(2001)129<0420:ASWTET>2.0.CO;2)). It is a form of ensemble square-root inversion, and was previously implemented in [Huang et al., 2022](http://doi.org/10.1088/1361-6420/ac99fa). The main advantage of ETKI over EKI is that it has better scalability as the observation dimension grows: while the naive implementation of EKI scales as ``\mathcal{O}(p^3)`` in the observation dimension ``p``, ETKI scales as ``\mathcal{O}(p)``. This, however, refers to the online cost. ETKI may have an offline cost of ``\mathcal{O}(p^3)`` if ``\Gamma`` is not easily invertible; see below.

The major disadvantage of ETKI is that it cannot be used with localization or sampling error correction. 

!!! note "Creating scalable observational covariances"
    ETKI requires storing and inverting the observation noise covariance, ``\Gamma^{-1}``. Without care, this can be prohibitively expensive. To this end, we have tools and an API for creating and using scalable or compact representations of covariances that are necessary for scalability. See [here](@ref building-covariances) for details and examples. 
## Using ETKI

An ETKI struct can be created using the `EnsembleKalmanProcess` constructor by specifying the `TransformInversion` process type: 

```julia
using EnsembleKalmanProcesses
# given the prior distribution `prior`, data `y` and covariance `obs_noise_cov`,

J = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior

etkiobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov,
                               TransformInversion())
```

The rest of the inversion process is the same as for regular EKI.

# [Sparsity-Inducing Ensemble Kalman Inversion](@id seki)

We include Sparsity-inducing Ensemble Kalman Inversion (SEKI) to add approximate ``L^0`` and ``L^1`` penalization to the EKI ([Schneider, Stuart, Wu, 2020](https://doi.org/10.48550/arXiv.2007.06175)).

!!! warning
    The algorithm suffers from robustness issues, and therefore we urge caution in using the tool

