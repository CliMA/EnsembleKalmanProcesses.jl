# [Ensemble Kalman Sampling](@id eks)

### What Is It and What Does It Do?
The Ensemble Kalman Sampler (EKS) ([Garbuno-Inigo et al, 2020](https://doi.org/10.1137/19M1251655), [Cleary et al, 2020](https://doi.org/10.1016/j.jcp.2020.109716), [Garbuno-Inigo et al, 2020](https://doi.org/10.1137/19M1304891)) is a derivative-free tool for approximate Bayesian inference. It does so by approximately sampling from the posterior distribution. That is, EKS provides both point estimation (through the mean of the final ensemble) and uncertainty quantification (through the covariance of the final ensemble), this is in contrast to EKI, which only provides point estimation.

The EKS is an interacting particle system in stochastic differential equation form, and it is based on a dynamic which transforms an arbitrary initial probability distribution into an approximation of the desired posterior distribution over an infinite time horizon -- see [Garbuno-Inigo et al, 2020](https://doi.org/10.1137/19M1251655), for a comprehensive description of the method. While there are noisy variants of the standard EKI, EKS differs from them in its noise structure (as its noise is added in parameter space, not in  data space), and its update rule explicitly accounts for the prior (rather than having it enter through initialization). The EKS algorithm can be understood as well as an affine invariant system of interacting particles ([Garbuno-Inigo et al, 2020](https://doi.org/10.1137/19M1304891)) for which a finite-sample correction is introduced to overcome its computational finite-sample implementation. This finite-sample corrected version of EKS is also known as the affine invariant Langevin dynamics (ALDI).


Perhaps as expected, the approximate posterior characterization through EKS needs more iterations, and thus more forward model evaluations, than EKI to converge to a suitable solution. This is because of the discrete-time implementation of the EKS diffusion process and the need to maintain a stable interacting particle system. However, the posterior approximation through EKS is obtained with far less computational effort than a typical Markov Chain Monte Carlo (MCMC) like Metropolis-Hastings.

### Problem Formulation

The data ``y`` and parameter vector ``\theta`` are assumed to be related according to:
```math
    y = \mathcal{G}(\theta) + \eta \,,
```
where ``\mathcal{G}:  \mathbb{R}^p \rightarrow \mathbb{R}^d`` denotes the forward map, ``y \in \mathbb{R}^d`` is the vector of observations, and ``\eta`` is the observational noise, which is assumed to be drawn from a ``d``-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``. The objective of the inverse problem is to compute the unknown parameters ``\theta`` given the observations ``y``, the known forward map ``\mathcal{G}``, and noise characteristics ``\eta`` of the process.

!!! note
    To obtain Bayesian characterization for the posterior from EKS, the user must specify a Gaussian prior distribution. See [Prior distributions](@ref parameter-distributions) to see how one can apply flexible constraints while maintaining Gaussian priors. 


### Ensemble Kalman Sampling Algorithm


The EKS is based on the following update equation for the parameter vector ``\theta^{(j)}_n`` of ensemble member ``j`` at the ``n``-iteration:

```math
\begin{aligned}
\theta_{n+1}^{(*, j)} &= \theta_{n}^{(j)} - \dfrac{\Delta t_n}{J}\sum_{k=1}^J\langle \mathcal{G}(\theta_n^{(k)}) - \bar{\mathcal{G}}_n, \Gamma_y^{-1}(\mathcal{G}(\theta_n^{(j)}) - y) \rangle \theta_{n}^{(k)} + \frac{d+1}{J} \left(\theta_{n}^{(j)} - \bar \theta_n \right) - \Delta t_n \mathsf{C}(\Theta_n) \Gamma_{\theta}^{-1} \theta_{n + 1}^{(*, j)} \,, \\
\theta_{n + 1}^{j} &= \theta_{n+1}^{(*, j)} + \sqrt{2 \Delta t_n \mathsf{C}(\Theta_n)} \xi_n^{j} \,,
\end{aligned}
```

where the subscript ``n=1, \dots, N_{\text{it}}`` indicates the iteration, ``J`` is the ensemble size (i.e., the number of particles in the ensemble), ``\Delta t_n`` is an internal adaptive time step (thus no need for the user to specify), ``\Gamma_{\theta}`` is the prior covariance, and ``\xi_n^{(j)} \sim \mathcal{N}(0, \mathrm{I}_p)``. ``\bar{\mathcal{G}}_n`` is the ensemble mean of the forward map ``\mathcal{G}(\theta)``,

```math
\bar{\mathcal{G}}_n = \dfrac{1}{J}\sum_{k=1}^J\mathcal{G}(\theta_n^{(k)})\,.
```

The ``p \times p`` matrix ``\mathsf{C}(\Theta_n)``, where ``\Theta_n = \left\{\theta^{(j)}_n\right\}_{j=1}^{J}`` is the set of all ensemble particles in the ``n``-th iteration, denotes the empirical covariance between particles

```math
\mathsf{C}(\Theta_n) = \frac{1}{J} \sum_{k=1}^J (\theta^{(k)}_n - \bar{\theta}_n) \otimes (\theta^{(k)}_n - \bar{\theta}_n)\,,
```
where ``\bar{\theta}_n`` is the ensemble mean of the particles,

```math
\bar{\theta}_n = \dfrac{1}{J}\sum_{k=1}^J\theta^{(k)}_n \,.
```

### Constructing the Forward Map

At the core of the forward map ``\mathcal{G}`` is the dynamical model ``\Psi:\mathbb{R}^p \rightarrow \mathbb{R}^o`` (running ``\Psi`` is usually where the computational heavy-lifting is done), but the map ``\mathcal{G}`` may include additional components such as a transformation of the (unbounded) parameters ``\theta`` to a constrained domain the dynamical model can work with, or some post-processing of the output of ``\Psi`` to generate the observations. For example, ``\mathcal{G}`` may take the following form:

```math
\mathcal{G} = \mathcal{H} \circ \Psi \circ \mathcal{T}^{-1},
```
where ``\mathcal{H}:\mathbb{R}^o \rightarrow \mathbb{R}^d`` is the observation map and ``\mathcal{T}`` is the transformation from the constrained to the unconstrained parameter space, such that ``\mathcal{T}(\phi)=\theta``. A family of standard transformations and their inverses are available in the `ParameterDistributions` module.


### How to Construct an Ensemble Kalman Sampler

An EKS object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Sampler` type. The constructor takes two arguments, the prior mean `prior_mean` and the prior covariance `prior_cov`.

Creating an EKS object requires as arguments:

 1. An initial parameter ensemble -- an array of size `p × N_ens`, where `N_ens` is the  ensemble size;

 2. The mean value of the observed data -- a vector of length `d`;

 3. The covariance matrix of the observational noise -- an array of size `d × d`;

 4. The `Sampler(prior_mean, prior_cov)` process type, with the mean (a vector of length `p`) and the covariance (an array of size `p x p`) of the parameter's prior distribution.

The following example shows how an EKS object is instantiated. An observation (`y`) and the covariance of the observational noise (`obs_cov`) are assumed to be defined previously in the code.

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions  # required to create the prior

# Construct prior (see `ParameterDistributions.jl` docs)
prior = ParameterDistribution(...)
prior_mean = mean(prior)
prior_cov = cov(prior)

# Construct initial ensemble
N_ens = 50  # ensemble size
initial_ensemble = construct_initial_ensemble(prior, N_ens)

# Construct ensemble Kalman process
eks_process = Sampler(prior_mean, prior_cov)
eksobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, eks_process)
```


### Updating the ensemble

Once the EKS object `eksobj` has been initialized, the initial ensemble of particles is iteratively updated by the `update_ensemble!` function, which takes as arguments the `eksobj` and the evaluations of the forward model at each member of the current ensemble. In the following example, the forward map `G` maps a parameter to the corresponding data -- this is done for each parameter in the ensemble, such that the resulting `g_ens` is of size `d x N_ens`. The `update_ensemble!` function then stores the updated ensemble as well as the evaluations of the forward map in `eksobj`.

A typical use of the `update_ensemble!` function given the EKS object `eksobj`, the dynamical model `Ψ`, and the observation map `H` (the latter two are assumed to be defined elsewhere, e.g. in a separate module)  may look as follows:


```julia
# Given:
# Ψ (some black box simulator)
# H (some observation of the simulator output)
# prior (prior distribution and parameter constraints)

N_iter = 100 # Number of iterations

for n in 1:N_iter
    ϕ_n = get_ϕ_final(prior, eksobj) # Get current ensemble in constrained "ϕ"-space
    G_n = [H(Ψ(ϕ_n[:, i])) for i in 1:J]  # Evaluate forward map
    g_ens = hcat(G_n...)  # Reformat into `d x N_ens` matrix
    update_ensemble!(eksobj, g_ens) # Update ensemble
end
```

### Solution

The solution of the EKS algorithm is an approximate Gaussian distribution whose mean (`θ_post`) and covariance (`Γ_post`) can be extracted from the ''final ensemble'' (i.e., after the last iteration). The sample mean of the last ensemble is also the "optimal" parameter (`θ_optim`) for the given calibration problem. These statistics can be accessed as follows:


```julia
# mean of the Gaussian distribution, the optimal parameter in computational θ-space
θ_post = get_u_mean_final(eksobj)
# (empirical) covariance of the Gaussian distribution in computational θ-space
Γ_post = get_u_cov_final(eksobj)
```
To obtain samples of this approximate posterior in the constrained space, we first sample the distribution, then transform using the constraints contained within the prior 
```julia
using Random, Distributions

ten_post_samples = rand(MvNormal(θ_post,Γ_post), 10)
ten_post_samples_phys = transform_unconstrained_to_constrained(prior, ten_post_samples) # the optimal physical parameter value
```
