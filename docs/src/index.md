# EnsembleKalmanProcesses

`EnsembleKalmanProcesses.jl` (EKP) is a library of derivative-free Bayesian optimization techniques based on the Ensemble Kalman Filters, a well known family of approximate filters used for data assimilation. Currently, the following methods are implemented in the library:
 - Ensemble Kalman Inversion (EKI) - The traditional optimization technique based on the Ensemble Kalman Filter EnKF ([Iglesias, Law, Stuart, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)),
 - Ensemble Kalman Sampler (EKS) - also obtains a Gaussian Approximation of the posterior distribution, through a Monte Carlo integration ([Garbuno-Inigo, Hoffmann, Li, Stuart, 2020](https://doi.org/10.1137/19M1251655)),
 - Unscented Kalman Inversion (UKI) - also obtains a Gaussian Approximation of the posterior distribution, through a quadrature based integration approach (Huang, Schneider, Stuart, 2020),
 - [coming soon] Sparsity preserving Ensemble Kalman Inversion (SEKI) - Additionally adds approximate ``L^0`` and ``L^1`` penalization to the EKI (Schneider, Stuart, Wu, 2020).

Module                                      | Purpose
--------------------------------------------|--------------------------------------------------------
EnsembleKalmanProcesses.jl                  | Collection of all tools
EnsembleKalmanProcess.jl                    | Implementations of EKI, EKS and UKI
Observations.jl                             | Structure to hold observational data
ParameterDistributions.jl                   | Structures to hold prior and posterior distributions
DataStorage.jl                              | Structure to hold model parameters and outputs
