# Ensemble Kalman Sampling

### What Is It and What Does It Do?
Ensemble Kalman Sampling ([Garbuno-Inigo et al, 2019](https://arxiv.org/pdf/1903.08866.pdf), [Cleary et al, 2020](https://clima.caltech.edu/files/2020/01/2001.03689.pdf)) is a derivative-free optimization method that can be used to solve the inverse problem of finding the optimal model parameters given noisy data. In contrast to ensemble Kalman inversion ([Iglesias et al, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)), whose iterative updates result in a collapse of the ensemble onto the optimal parameter, the ensemble Kalman sampler generates approximate samples from the Bayesian posterior distribution of the parameter -- i.e., it can be used not only for point estimation of the optimal parameter (as provided by the mean of the particles after the last iteration), but also for (approximative) uncertainty quantification (as provided by the covariance of the particles after the last iteration). 


The ensemble Kalman sampler is an interacting particle system in stochastic differential equation form, and it is based on a dynamic which transforms an arbitrary initial distribution into the desired posterior distribution, over an infinite time horizon -- see [Garbuno-Inigo et al, 2019](https://arxiv.org/pdf/1903.08866.pdf), for a comprehensive description of the method. The ensemble Kalman sampling algorithm results from the introduction of a (judiciously chosen) noise to the ensemble Kalman inversion algorithm. Note that while there are also noisy variants of the standard ensemble Kalman inversion, ensemble Kalman sampling differs from them in its noise structure (its noise is added in parameter space, not in  data space), and its update rule explicitly accounts for the prior (rather than having it enter through initialization).


### Problem Formulation

The data ``y`` and parameter vector ``\theta`` are assumed to be related according to:
```math
    y = \mathcal{G}(\theta) + \eta, \,
```
where ``\mathcal{G}:  \mathbb{R}^p \rightarrow \mathbb{R}^d`` denotes the
forward model, ``y \in \mathbb{R}^d`` is the vector of observations, and ``\eta`` is the observational noise, which is assumed to be drawn from a d-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``. The objective of the inverse problem is to compute the unknown model parameters ``\theta`` given the observations ``y``, the known forward model ``\mathcal{G}``, and noise characteristics $\eta$ of the process.


### How to Construct an Ensemble Kalman Sampler

An ensemble Kalman sampling object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Sampler(prior_mean, prior_cov)` process type.

Creating an ensemble Kalman inversion object requires as arguments:

 1. An initial parameter ensemble -- an array of size `p × N_ens`, where `N_ens` is the  ensemble size;
 
 2. The mean value of the observed data -- a vector of length `d`;
 
 3. The covariance matrix of the observational noise -- an array of size `d × d`;
 
 4. The `Sampler(prior_mean, prior_cov)` process type, with the mean (a vector of length `p`) and the covariance (an array of size `p x p`) of the parameter's prior distribution

The following example shows how an ensemble Kalman sampling object is instantiated. The mean of the observational data (`obs_mean`) and the covariance of the observational noise (`obs_cov`) are assumed to be defined previously in the code.

```julia
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using Distributions

# Construct priors for parameters "A" and "B" (see `ParameterDistributionStorage` docs)
p1 = "A"  # parameter name
c1 = no_constraint() # A has no constraints
d1 = Parameterized(Normal(0, 1))  # A has a standard normal distribution in prior space
p2 = "B"  # parameter name
c2 = no_constraint()  # B has no constraints
d2 = Parameterized(Normal(0, 1))  # B has a standard normal distribution in prior space

distributions = [d1, d2]
constraints = [[c1], [c2]]
names = [p1, p2] 
prior = ParameterDistribution(distributions, constraints, names)
prior_mean = get_mean(prior)
prior_cov = get_cov(prior)

# Construct initial ensemble
N_ens = 50  # number of ensemble members
initial_ensemble = construct_initial_ensemble(prior, N_ens) 

# Construct ensemble Kalman process 
eks_process = Sampler(prior_mean, prior_cov)
eks_obj = EnsembleKalmanProcess(initial_ensemble, obs_mean, obs_noise_cov, eks_process)
```

Note that no information about the forward model is necessary to instantiate the ensemble Kalman process. The forward model is only used in the process of updating the initial ensemble, where it maps the ensemble of particles (parameters) to the corresponding data.

### Updating the ensemble

Once the ensemble Kalman sampling object `eks_obj` has been initialized, the initial ensemble of particles is iteratively updated by the `update_ensemble!` function, which takes as arguments the `eks_obj` and the evaluations of the forward model at each member of the current ensemble. In the following example, the forward model `G_ens` (defined elsewhere, e.g. in a separate module) maps an array of `N_ens` parameters (size: `p × N_ens`) to the array of corresponding data (size: `d x N_ens`). The `update_ensemble!` function then stores the updated ensemble as well as the forward model evaluations in `eks_obj`.


```julia
N_iter = 20 # Number of steps of the algorithm

for i in 1:N_iter
    params_i = get_u_final(eks_obj)  # Get current ensemble
    g_ens = G_ens(params_i)  # Evaluate forward model
    update_ensemble!(eks_obj, g_ens)  # Update ensemble
end
```

### Solution

The ensemble Kalman sampling algorithm converts the initial ensemble of particles into approximate samples of the posterior distribution. Thus, the optimal parameter `theta_optim` found by the algorithm is given by the mean of the ''last ensemble'' (i.e., the ensemble after the last iteration), and the standard deviation of the last ensemble serves as a measure of the uncertainty of the optimal parameter. `theta_optim` and its standard deviation `sigma` can be accessed as follows: 

```julia
using Statistics

theta_optim = mean(get_u_final(eks_obj), dims=2)
sigma = cov(get_u_final(eks_obj), dims=2)
```
