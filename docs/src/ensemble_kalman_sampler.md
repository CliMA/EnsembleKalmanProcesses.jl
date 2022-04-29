# Ensemble Kalman Sampling

### What Is It and What Does It Do?
The Ensemble Kalman Sampler (EKS) ([Garbuno-Inigo et al, 2019](https://arxiv.org/pdf/1903.08866.pdf), [Cleary et al, 2020](https://clima.caltech.edu/files/2020/01/2001.03689.pdf), [Garbuno-Inigo et al, 2020](https://arxiv.org/pdf/1912.02859.pdf)) is a derivative-free method that can be used to solve the inverse problem of finding the optimal model parameters given noisy data. In contrast to Ensemble Kalman Inversion (EKI) ([Iglesias et al, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)), the EKS method approximately samples from the posterior distribution; that is, EKS provides both point estimation (through the mean of the final ensemble) and uncertainty quantification (through the covariance of the final ensemble), unlike EKI, which only provides the former.


The EKS is an interacting particle system in stochastic differential equation form, and it is based on a dynamic which transforms an arbitrary initial probability distribution into an approximation of the desired posterior distribution over an infinite time horizon -- see [Garbuno-Inigo et al, 2019](https://arxiv.org/pdf/1903.08866.pdf), for a comprehensive description of the method. While there are noisy variants of the standard EKI, EKS differs from them in its noise structure (as its noise is added in parameter space, not in  data space), and its update rule explicitly accounts for the prior (rather than having it enter through initialization). The EKS algorithm can be understood as well as an affine invariant system of interacting particles ([Garbuno-Inigo et al, 2020](https://arxiv.org/pdf/1912.02859.pdf)) for which a finite-sample correction is introduced to overcome its computational finite-sample implementation. The finite-sample corrected version of EKS is referred to as ALDI for its acronym in ([Garbuno-Inigo et al, 2020](https://arxiv.org/pdf/1912.02859.pdf)). 


Note that in practice the approximate posterior characterization through EKS needs more iterations, and thus more forward model evaluations, than EKI. This is because of the discrete-time implementation of the EKS diffusion process and the need to maintain a stable interacting particle system. However, the posterior approximation through EKS is obtained with less computational effort than a typical Markov Chain Monte Carlo (MCMC) like Metropolis-Hastings.

### Problem Formulation

The data ``y`` and parameter vector ``\theta`` are assumed to be related according to:
```math
    y = \mathcal{G}(\theta) + \eta \,,
```
where ``\mathcal{G}:  \mathbb{R}^p \rightarrow \mathbb{R}^d`` denotes the forward map, ``y \in \mathbb{R}^d`` is the vector of observations, and ``\eta`` is the observational noise, which is assumed to be drawn from a ``d``-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``. The objective of the inverse problem is to compute the unknown parameters ``\theta`` given the observations ``y``, the known forward map ``\mathcal{G}``, and noise characteristics ``\eta`` of the process. The full Bayesian characterization for the posterior under the EKS framework requires a ``p``-dimensional prior Gaussian distribution ``\mathcal{N}(m_\theta, \Gamma_\theta)``.


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

The following example shows how an EKS object is instantiated. The mean of the observational data (`obs_mean`) and the covariance of the observational noise (`obs_cov`) are assumed to be defined previously in the code.

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
eks_obj = EnsembleKalmanProcess(initial_ensemble, obs_mean, obs_noise_cov, eks_process)
```


### Updating the ensemble

Once the EKS object `eks_obj` has been initialized, the initial ensemble of particles is iteratively updated by the `update_ensemble!` function, which takes as arguments the `eks_obj` and the evaluations of the forward model at each member of the current ensemble. In the following example, the forward map `G` maps a parameter to the corresponding data -- this is done for each parameter in the ensemble, such that the resulting `g_ens` is of size `d x N_ens`. The `update_ensemble!` function then stores the updated ensemble as well as the evaluations of the forward map in `eks_obj`.

A typical use of the `update_ensemble!` function given the EKS object `eks_obj`, the dynamical model `Ψ`, and the observation map `H` (the latter two are assumed to be defined elsewhere, e.g. in a separate module)  may look as follows:


```julia
N_iter = 10 # Number of iterations

for n in 1:N_iter
    θ_n = get_u_final(eks_obj) # Get current ensemble
    ϕ_n = transform_unconstrained_to_constrained(prior, θ_n) # Transform parameters to physical/constrained space
    G_n = [H(Ψ(ϕ_n[:, i])) for i in 1:J]  # Evaluate forward map
    g_ens = hcat(G_n...)  # Reformat into `d x N_ens` matrix
    update_ensemble!(eks_obj, g_ens) # Update ensemble
end
```

### Solution

The solution of the EKS algorithm is an approximate Gaussian distribution whose mean (`θ_post`) and covariance (`Γ_post`) can be extracted from the ''last ensemble'' (i.e., the ensemble after the last iteration). The sample mean of the last ensemble is also the "optimal" parameter (`θ_optim`) for the given calibration problem. These statistics can be accessed as follows:


```julia
using Statistics

# mean of the Gaussian distribution, also the optimal parameter for the calibration problem
θ_post = mean(get_u_final(eks_obj), dims=2)
# covariance of the Gaussian distribution
Γ_post = cov(get_u_final(eks_obj), dims=2)
```
