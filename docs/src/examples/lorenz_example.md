# [Lorenz 96 example](@id Lorenz-example)

!!! info "How do I run this code?"
    The full code is found in the [`examples/`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples) directory of the github repository

## Overview

The Lorenz 96 (hereafter L96) example is a toy-problem for the application of the EnsembleKalmanProcesses.jl optimization and approximate uncertainty quantification methodologies.
Here is L96 with additional periodic-in-time forcing, we try to determine parameters (sinusoidal amplitude and stationary component of the forcing) from some output statistics.
The standard L96 equations are implemented with an additional forcing term with time dependence.
The output statistics which are used for learning are the finite time-averaged variances.

## Lorenz 96 equations

The standard single-scale L96 equations are implemented.
The Lorenz 96 system ([Lorenz, 1996](http://www.raidl.cz/file/18/lorenz-1996-_predictability_partly_solved.pdf)) is given by 
```math
\frac{d x_i}{d t} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,
```
with $i$ indicating the index of the given longitude. The number of longitudes is given by $N$.
The boundary conditions are given by
```math
x_{-1} = x_{N-1}, \ x_0 = x_N, \ x_{N+1} = x_1.
```
The time scaling is such that the characteristic time is 5 days ([Lorenz, 1996](http://www.raidl.cz/file/18/lorenz-1996-_predictability_partly_solved.pdf)). 
For very small values of ``F``, the solutions $x_i$ decay to $F$ after the initial transient feature.
For moderate values of ``F``, the solutions are periodic, and for larger values of ``F``, the system is chaotic.
The solution variance is a function of the forcing magnitude.
Variations in the base state as a function of time can be imposed through a time-dependent forcing term ``F(t)``.

A temporal forcing term is defined
```math
F = F_s + A \sin(\omega t),
```
with steady-state forcing ``F_s``, transient forcing amplitude ``A``, and transient forcing frequency ``\omega``.
The total forcing ``F`` must be within the chaotic regime of L96 for all time given the prescribed $N$.

The L96 dynamics are solved with RK4 integration.


## Structure

The main code is located in `Lorenz_example.jl` which provides the functionality to run the L96 dynamical system, extract time-averaged statistics from the L96 states, and use the time-average statistics for optimization and uncertainty quantification.

The L96 system is solved in `GModel.jl` according to the time integration settings specified in `LSettings` and the L96 parameters specified in `LParams`.
The types of statistics to be collected are detailed in `GModel.jl`.


## Lorenz dynamics inputs

### Dynamics settings
The use of the transient forcing term is with the flag, `dynamics`. Stationary forcing is `dynamics=1` ($A=0$) and transient forcing is used with `dynamics=2` ($A\neq0$).
The default parameters are specified in `Lorenz_example.jl` and can be modified as necessary.
The system is solved over time horizon $0$ to `tend` at fixed time step `dt`.
```julia
N = 36
dt = 1/64
t_start = 800
```

### Inverse problem settings
The states are integrated over time `Ts_days` to construct the time averaged statistics for use by the optimization.
The specification of the statistics to be gathered from the states are provided by `stats_type`.
The Ensemble Kalman Process (EKP) settings are
```julia
N_ens = 20 # number of ensemble members
N_iter = 5 # number of EKI iterations
```


## Setting up the Inverse Problem
The goal is to learn ```F_s``` and ```A``` based on the time averaged statistics in a perfect model setting.
The true parameters are
```julia
F_true = 8. # Mean F
A_true = 2.5 # Transient F amplitude
ω_true = 2. * π / (360. / τc) # Frequency of the transient F
params_true = [F_true, A_true]
param_names = ["F", "A"]
```

### Priors
We implement (biased) priors as follows

```julia
prior_means = [F_true + 1.0, A_true + 0.5]
prior_stds = [2.0, 0.5 * A_true]
# constrained_gaussian("name", desired_mean, desired_std, lower_bd, upper_bd)
prior_F = constrained_gaussian(param_names[1], prior_means[1], prior_stds[1], 0, Inf)
prior_A = constrained_gaussian(param_names[2], prior_means[2], prior_stds[2], 0, Inf)
priors = combine_distributions([prior_F, prior_A])
```
We use the recommended [`constrained_gaussian`](@ref constrained-gaussian) to add the desired scale and bounds to the prior distribution, in particular we place lower bounds to preserve positivity. 

### Observational Noise
The observational noise can be generated using the L96 system or prescribed, as specified by `var_prescribe`. 

`var_prescribe==false`
The observational noise is constructed by generating independent instantiations of the L96 statistics of interest at the true parameters for different initial conditions.
The empirical covariance matrix is constructed.

`var_prescribe==true`
The observational noise is prescribed as a Gaussian distribution with prescribed mean and variance.

## Running the Example
The L96 parameter estimation can be run using `julia --project Lorenz_example.jl`


## Solution and Output
The output will provide the estimated parameters in the constrained `ϕ`-space. The `priors` are required in the get-method to apply these constraints.

### Printed output
```julia
# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)
println("\nEKI results:")
println(get_ϕ_mean_final(priors, ekiobj))
```

### Saved output
The parameters and forward model outputs will be saved in `parameter_storage.jld2` and `data_storage.jld2`, respectively.
The data will be saved in the directory `output`.

### Plots
A scatter plot animation of the ensemble convergence to the true parameters is saved in the directory `output`.
