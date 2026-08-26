# [Gauss Newton Kalman Inversion](@id gnki)

### What Is It and What Does It Do?
Gauss Newton Kalman Inversion (GNKI) [Chada21a, Chen13c](@citep), also known as the Iterative Ensemble Kalman Filter with Statistical Linearization, is a derivative-free ensemble optimization method based on the Gauss Newton optimization update and the Iterative Extended Kalman Filter (IExKF) [Jazwinski70a](@citep).  In the linear case and continuous limit, GNKI recovers the true posterior mean and covariance.  Empirically, GNKI performs well as an optimization algorithm in the nonlinear case.  

### Problem Formulation

The data ``y`` and parameter vector ``\theta`` are assumed to be related according to:
```math
\tag{1} y = \mathcal{G}(\theta) + \eta \,,
```
where ``\mathcal{G}:  \mathbb{R}^p \rightarrow \mathbb{R}^d`` denotes the forward map, ``y \in \mathbb{R}^d`` is the vector of observations, and ``\eta`` is the observational noise, which is assumed to be drawn from a ``d``-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``. The objective of the inverse problem is to compute the unknown parameters ``\theta`` given the observations ``y``, the known forward map ``\mathcal{G}``, and noise characteristics ``\eta`` of the process.

!!! note
    GNKI relies on minimizing a loss function that includes regularization.  The user must specify a Gaussian prior with distribution ``\mathcal{N}(m, \Gamma_{\theta})``. See [Prior distributions](@ref parameter-distributions) to see how one can apply flexible constraints while maintaining Gaussian priors. 

The optimal parameters ``\theta^*`` given relation (1) minimize the loss 

 ```math
\mathcal{L}(\theta, y) = \frac{1}{2} \langle \mathcal{G}(\theta) - y \, , \, \Gamma_y^{-1} \left ( \mathcal{G}(\theta) - y \right ) \rangle + \frac{1}{2} \langle m - \theta \, , \, \Gamma_{\theta}^{-1} \left ( m - \theta  \right ) \rangle,
```

where ``m`` is the prior mean and ``\Gamma_{\theta}`` is the prior covariance. 

### Algorithm

GNKI updates the ``j``-th ensemble member at the ``n``-th iteration by directly approximating the Jacobian with statistics from the ensemble.

First, the ensemble covariance matrices are computed: 
```math
\begin{aligned}
        &\mathcal{G}_n^{(j)}  = \mathcal{G}(\theta_n^{(j)}) \qquad 
        \bar{\mathcal{G}}_n = \dfrac{1}{J}\sum_{k=1}^J\mathcal{G}_n^{(k)} \\
        & C^{\theta \mathcal{G}}_n = \dfrac{1}{J - 1}\sum_{k=1}^{J}
        (\theta_n^{(k)} - \bar{\theta}_n )(\mathcal{G}_n^{(k)} - \bar{\mathcal{G}}_n)^T \\
        & C^{\theta \theta}_n = \dfrac{1}{J - 1} \sum_{k=1}^{J} 
        (\theta_n^{(k)} - \bar{\theta}_n )(\theta_n^{(k)} - \bar{\theta}_n )^T.

\end{aligned}
```

Using the ensemble covariance matrices, the update equation from ``n`` to ``n+1`` under GNKI is
```math
\begin{aligned}
        & \theta_{n+1}^{(j)} = \theta_n^{(j)} + \alpha \left\{ K_n\left(y_n^{(j)} - \mathcal{G}(\theta_n^{(j)})\right) + \left(I - K_n G_n\right)\left(m_n^{(j)} - \theta_n^{(j)}\right) \right\} \\
        
        & \\

        & K_n = \Gamma_{\theta} G_n^T \left(G_n \Gamma_{\theta} G_n^T + \Gamma_{y}\right)^{-1} \\
        
        & G_n = \left(C^{\theta \mathcal{G}}_n\right)^T \left(C^{\theta \theta}_n\right)^{-1}, 


\end{aligned}
```

where ``y_n^{(j)} \sim \mathcal{N}(y, 2\alpha^{-1}\Gamma_y)`` and ``m_n^{(j)} \sim \mathcal{N}(m, 2\alpha^{-1}\Gamma_{\theta})``, and where ``\alpha = \Delta t_n`` is the timestep from the [learning rate scheduler](@ref learning-rate-schedulers).

!!! note "Defaults"
    By default, GNKI is constructed with the `NesterovAccelerator()` accelerator, `SECNice()` localization, and the `DefaultScheduler()`; see the [defaults](@ref defaults) page.

## Creating the GNKI Object

We first build a prior distribution (for details of the construction see [here](@ref constrained-gaussian)). 
Then we build our EKP object with 
```julia
using EnsembleKalmanProcesses

gnkiobj = EnsembleKalmanProcess(args..., GaussNewtonInversion(prior); kwargs...)
```
For general EKP object creation requirements and for how to make updates using the inversion algorithm, see the [Ensemble Kalman Inversion](@ref eki) page.



