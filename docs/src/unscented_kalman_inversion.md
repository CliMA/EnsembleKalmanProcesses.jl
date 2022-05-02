# Unscented Kalman Inversion

One of the ensemble Kalman processes implemented in `EnsembleKalmanProcesses.jl` is the unscented Kalman inversion ([Huang et al, 2021](https://arxiv.org/abs/2102.01580)). The unscented Kalman inversion (UKI) is a derivative-free ensemble optimization method that seeks to find the optimal parameters ``\theta \in \mathbb{R}^p`` in the inverse problem
```math
 y = \mathcal{G}(\theta) + \eta
```
where ``\mathcal{G}`` denotes the forward map, ``y \in \mathbb{R}^d`` is the vector of observations and ``\eta \sim \mathcal{N}(0, \Gamma_y)`` is additive Gaussian observational noise. Note that ``p`` is the size of the parameter vector ``\theta`` and ``d`` is taken to be the size of the observation vector ``y``. The UKI algorithm has the following properties

* UKI has a given ensemble size and requires only ``2p + 1`` particles in general.
* UKI has uncertainty quantification capability, it gives both mean and covariance approximation (no ensemble collapse and no empirical variance inflation) of the posterior distribution, the 3-sigma confidence interval covers the truth parameters for perfect models.

## Algorithm
 
The UKI applies unscented Kalman filter for the following stochastic dynamical system 

```math
\begin{aligned}
  &\textrm{evolution:}    &&\theta_{n+1} = r + \alpha (\theta_{n}  - r) +  \omega_{n+1}, &&\omega_{n+1} \sim \mathcal{N}(0,\Sigma_{\omega}),\\
  &\textrm{observation:}  &&y_{n+1} = \mathcal{G}(\theta_{n+1}) + \nu_{n+1}, &&\nu_{n+1} \sim \mathcal{N}(0,\Sigma_{\nu}).
\end{aligned}
```
The free parameters in the UKI are ``\alpha, r, \Sigma_{\nu}, \Sigma_{\omega}``.
The UKI updates both the mean ``m_n`` and covariance ``C_n`` estimations of the parameter vector ``\theta`` as following

* Prediction step :

```math
\begin{aligned}
    \hat{m}_{n+1} = & r+\alpha(m_n-r)\\
    \hat{C}_{n+1} = & \alpha^2 C_{n} + \Sigma_{\omega}
\end{aligned}
```  
* Generate sigma points :
```math    
\begin{aligned}
    &\hat{\theta}_{n+1}^0 = \hat{m}_{n+1} \\
    &\hat{\theta}_{n+1}^j = \hat{m}_{n+1} + c_j [\sqrt{\hat{C}_{n+1}}]_j \quad (1\leq j\leq N_\theta)\\ 
    &\hat{\theta}_{n+1}^{j+N_\theta} = \hat{m}_{n+1} - c_j [\sqrt{\hat{C}_{n+1}}]_j\quad (1\leq j\leq N_\theta)
\end{aligned}
```     
*  Analysis step :
    
```math
   \begin{aligned}
        &\hat{y}^j_{n+1} = \mathcal{G}(\hat{\theta}^j_{n+1}) \qquad \hat{y}_{n+1} = \hat{y}^0_{n+1}\\
         &\hat{C}^{\theta p}_{n+1} = \sum_{j=1}^{2N_\theta}W_j^{c}
        (\hat{\theta}^j_{n+1} - \hat{m}_{n+1} )(\hat{y}^j_{n+1} - \hat{y}_{n+1})^T \\
        &\hat{C}^{pp}_{n+1} = \sum_{j=1}^{2N_\theta}W_j^{c}
        (\hat{y}^j_{n+1} - \hat{y}_{n+1} )(\hat{y}^j_{n+1} - \hat{y}_{n+1})^T + \Sigma_{\nu}\\
        &m_{n+1} = \hat{m}_{n+1} + \hat{C}^{\theta p}_{n+1}(\hat{C}^{pp}_{n+1})^{-1}(y - \hat{y}_{n+1})\\
        &C_{n+1} = \hat{C}_{n+1} - \hat{C}^{\theta p}_{n+1}(\hat{C}^{pp}_{n+1})^{-1}{\hat{C}^{\theta p}_{n+1}}{}^{T}\\
    \end{aligned}
```

The unscented transformation parameters are
```math
    \begin{aligned}
    &c_j = \sqrt{N_\theta +\lambda} \qquad W_j^{c} = \frac{1}{2(N_\theta+\lambda)}~(j=1,\cdots,2N_{\theta}).\\
    &\lambda = a^2 (N_\theta + \kappa) - N_\theta \quad a=\min\{\sqrt{\frac{4}{N_\theta + \kappa}},  1\}\quad  \kappa = 0\\
    \end{aligned}
```
And ``[\sqrt{C}]_j`` is the ``j``-th column of the Cholesky factor of ``C``. 
    
    


## Choice of free parameters
The free parameters in the unscented Kalman inversion are ``\alpha, r, \Sigma_{\nu}, \Sigma_{\omega}``, which are chosen based on theorems developed in [Huang et al, 2021](https://arxiv.org/abs/2102.01580)

* the vector ``r`` is set to be the prior mean

* the scalar ``\alpha \in (0,1]`` is a regularization parameter, which is used to overcome ill-posedness and overfitting. A practical guide is 

    * When the observation noise is negligible, and there are more observations than parameters (identifiable inverse problem) ``\alpha = 1`` (no regularization)
    * Otherwise ``\alpha < 1``. The smaller ``\alpha`` is, the closer UKI will converge to the prior mean.
    
* the matrix ``\Sigma_{\nu}`` is the artificial observation error covariance. We set ``\Sigma_{\nu} = 2 \Gamma_{y}``, which makes the inverse problem consistent. 

* the matrix ``\Sigma_{\omega}`` is the artificial evolution error covariance. We set ``\Sigma_{\omega} = (2 - \alpha^2)\Lambda``. We choose ``\Lambda`` as following

    * when there are more observations than parameters (identifiable inverse problem), ``\Lambda = C_n``, which is updated as the estimated covariance ``C_n`` in the ``n``-th every iteration. This guarantees the converged covariance matrix is a good approximation to the posterior covariance matrix with an uninformative prior.
    
    * otherwise ``\Lambda = C_0``, this allows that the converged covariance matrix is a weighted average between the posterior covariance matrix with an uninformative prior and ``C_0``.

In a nutshell, users only need to change the ``\alpha`` (`α_reg`), and the frequency to update the ``\Lambda`` (`update_freq`). The user can first try `α_reg = 1.0` and `update_freq = 0`.


## Implementation

### Initialization
An unscented Kalman inversion object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Unscented()` process type.

Creating an ensemble Kalman inversion object requires as arguments:
 1. The mean value of the observed outputs, a vector of size `[d]`;
 2. The covariance of the observational noise, a matrix of size `[d × d]`;
 3. The `Unscented()` process type.

The initialization of the `Unscented()` process requires prior mean and prior covariance, and the the size of the observation `d`. And user defined hyperparameters 
`α_reg` and `update_freq`.
```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions


# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg =  1.0
# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 0

process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

```

Note that no information about the forward map is necessary to initialize the Inversion process. The only forward map information required by the inversion process consists of model evaluations at the ensemble elements, necessary to update the ensemble.


### Constructing the Forward Map

At the core of the forward map ``\mathcal{G}`` is the dynamical model ``\Psi:\mathbb{R}^p \rightarrow \mathbb{R}^o`` (running ``\Psi`` is usually where the computational heavy-lifting is done), but the map ``\mathcal{G}`` may include additional components such as a transformation of the (unbounded) parameters ``\theta`` to a constrained domain the dynamical model can work with, or some post-processing of the output of ``\Psi`` to generate the observations. For example, ``\mathcal{G}`` may take the following form:

```math
\mathcal{G} = \mathcal{H} \circ \Psi \circ \mathcal{T}^{-1},
```
where ``\mathcal{H}:\mathbb{R}^o \rightarrow \mathbb{R}^d`` is the observation map and ``\mathcal{T}`` is the transformation from the constrained to the unconstrained parameter space, such that ``\mathcal{T}(\phi)=\theta``. A family of standard transformations and their inverses are available in the `ParameterDistributions` module.



### Updating the Ensemble

Once the unscented Kalman inversion object `UKIobj` has been initialized, any number of updates can be performed using the inversion algorithm.

A call to the inversion algorithm can be performed with the `update_ensemble!` function. This function takes as arguments the `UKIobj` and the evaluations of the forward map at each element of the current ensemble. The `update_ensemble!` function then stores the new updated ensemble and the inputted forward map evaluations in `UKIobj`.

The forward map ``\mathcal{G}`` maps the space of unconstrained parameters ``\theta`` to the outputs ``y \in \mathbb{R}^d``. In practice, the user may not have access to such a map directly. And the map is a composition of several functions. The `update_ensemble!` uses only the evalutaions `g_ens` but not the forward map  

For implementational reasons, the `update_ensemble` is performed by computing analysis stage first, followed by a prediction of the next sigma ensemble. And the first prediction is done in the initialization.

```julia
N_iter = 20 # Number of steps of the algorithm
 
for n in 1:N_iter

    # define black box parameter to observation map, 
    # with certain parameter transformation related to imposing some constraints
    # i.e. θ -> e^θ  -> G(e^θ) = y
    θ_n = get_u_final(ukiobj) # Get current ensemble
    ϕ_n = transform_unconstrained_to_constrained(prior, θ_n) # Transform parameters to physical/constrained space
    G_n = [H(Ψ(ϕ_n[:, i])) for i in 1:J]  # Evaluate forward map
    g_ens = hcat(G_n...)  # Reformat into `d x N_ens` matrix
    EnsembleKalmanProcesses.update_ensemble!(ukiobj, g_ens) # Update ensemble
end
```

### Solution

The solution of the unscented Kalman inversion algorithm is a Gaussian distribution whose mean and covariance can be extracted from the ''last ensemble'' (i.e., the ensemble after the last iteration). The sample mean of the last ensemble is also the "optimal" parameter (`θ_optim`) for the given calibration problem. These statistics can be accessed as follows: 

```julia
# mean of the Gaussian distribution, also the optimal parameter for the calibration problem
θ_optim = ukiobj.process.u_mean[end], 
# covariance of the Gaussian distribution
sigma = ukiobj.process.uu_cov[end]
```

There are two examples: [Lorenz96](../examples/Lorenz_example) and [Cloudy](../examples/Cloudy_example).
