# Ensemble Kalman Inversion

One of the ensemble Kalman processes implemented in `EnsembleKalmanProcesses.jl` is the ensemble Kalman inversion ([Iglesias et al, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)). The ensemble Kalman inversion (EKI) is a derivative-free ensemble optimization method that seeks to find the optimal parameters $\theta \in \mathbb{R}^p$ in the inverse problem

\\[ y = \mathcal{G}(\theta) + \eta, \\]

where $\mathcal{G}$ denotes the forward map, $y \in \mathbb{R}^d$ is the vector of observations and $\eta \sim \mathcal{N}(0, \Gamma_y)$ is additive Gaussian observational noise. Note that $p$ is the size of the parameter vector $\theta$ and $d$ is taken to be the size of the observation vector $y$. The EKI update equation for parameter vector $\theta^{(j)}$ of ensemble element $j$ is

```math
\theta_{n+1}^{(j)} = \theta_{n}^{(j)} - \dfrac{\Delta t_n}{J}\sum_{k=1}^J\langle \mathcal{G}(\theta_n^{(k)}) - \bar{\mathcal{G}}, \Gamma_y^{-1}(\mathcal{G}(\theta_n^{(j)}) - y) \rangle \theta_{n}^{(k)},
```

where the subscript $n$ indicates the iteration, $J$ is the number of elements in the ensemble and $\bar{\mathcal{G}}$ is the mean value of $\mathcal{G}(\theta)$ across ensemble elements,

\\[ \bar{\mathcal{G}} = \dfrac{1}{J}\sum_{k=1}^J\mathcal{G}(\theta^{(k)}). \\]

For typical applications, a near-optimal solution $\theta$ can be found after as few as 10 iterations of the algorithm. The obtained solution is optimal in the sense of the mean squared error loss, details can be found in [Iglesias et al (2013)](http://dx.doi.org/10.1088/0266-5611/29/4/045001). The algorithm performs better with larger ensembles. As a rule of thumb, the number of elements in the ensemble should be larger than $10p$, although the optimal ensemble size may depend on the problem setting and the computational power available.

An ensemble Kalman inversion object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Inversion()` process type.

Creating an ensemble Kalman inversion object requires as arguments:
 1. An initial parameter ensemble, `Array{FT, 2}` of size `[p × J]`;
 2. The mean value of the observed outputs, a vector of size `[d]`;
 3. The covariance of the observational noise, a matrix of size `[d × d]`
 4. The `Inversion()` process type.

A typical initialization of the `Inversion()` process may be in terms of a user-defined `prior`, a summary of the observation statistics given by the mean `y` and covariance `obs_noise_cov`, and a desired number of elements in the ensemble,
```julia
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

J = 50  # number of ensemble elements
initial_ensemble = construct_initial_ensemble(prior, J) # Initialize ensemble from prior

ekiobj = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Inversion())
```

Note that no information about the forward map is necessary to initialize the Inversion process. The only forward map information required by the inversion process consists of model evaluations at the ensemble elements, necessary to update the ensemble.

### Updating the Ensemble

Once the ensemble Kalman inversion object `ekiobj` has been initialized, any number of updates can be performed using the inversion algorithm.

A call to the inversion algorithm can be performed with the `update_ensemble!` function. This function takes as arguments the `ekiobj` and the evaluations of the forward map at each element of the current ensemble. The `update_ensemble!` function then stores the new updated ensemble and the inputted forward map evaluations in `ekiobj`.

A typical use of the `update_ensemble!` function given the ensemble Kalman inversion object `ekiobj` and the forward map `G` is
```julia
N_iter = 20 # Number of steps of the algorithm

for i in 1:N_iter
    params_i = get_u_final(ekiobj) # Get current ensemble
    g_ens = hcat([G(params_i[:,i]) for i in 1:J]...) # Evaluate forward map
    update_ensemble!(ekiobj, g_ens) # Update ensemble
end
```