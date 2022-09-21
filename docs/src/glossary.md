# Glossary

The following list includes the names and symbols of recurring concepts in
EnsembleKalmanProcesses.jl. Some of these variables do not appear in the codebase,
which relies on array programming for performance.  Contributions to the codebase
require following this notational convention. Similarly, if you find inconsistencies
in the documentation or codebase, please report an [issue](https://github.com/CliMA/EnsembleKalmanProcesses.jl/issues)
on GitHub.

| Name        | Symbol (Theory/Docs) | Symbol (Code)    |
| :---        |    :----:            |    :----:     |
| Parameter vector, Parameters (unconstrained space) | ``\theta``, ``u``, ``\mathcal{T}(\phi)``  | `θ`,`u` |
| Parameter vector, Parameters (physical / constrained space) | ``\phi``, ``\mathcal{T}^{-1}(\theta)``  | `ϕ`|
| Parameter vector size, Number of parameters  | ``p``  | `N_par` |
| Ensemble size  | ``J``  | `N_ens` |
| Ensemble particles, members  | ``\theta^{(j)}``  | |
| Number of iterations  | ``N_{\rm it}``  | `N_iter` |
| Observation vector, Observations, Data vector  | ``y``  | `y` |
| Observation vector size, Data vector size  | ``d``  | `N_obs` |
| Observational noise | ``\eta``  | obs_noise |
| Observational noise covariance | ``\Gamma_y``  | `obs_noise_cov` |
| Hilbert space inner product | ``\langle \phi , \Gamma^{-1} \psi \rangle`` | |
| Forward map | ``\mathcal{G}``  | `G` |
| Dynamical model | ``\Psi``  | `Ψ` |
| Transform map (constrained to unconstrained) | ``\mathcal{T}``  | `T` |
| Observation map | ``\mathcal{H}``  | `H` |
| Prior covariance (unconstrained space) | ``\Gamma_{\theta}``  | `prior_cov` |
| Prior mean (unconstrained space) | ``m_\theta``  | `prior_mean` |
