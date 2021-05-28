# Lorenz 96 example

#### Overview

The Lorenz 96 (hereafter L96) example is a toy-problem for the application of the EnsembleKalmanPRocesses.jl optimization and approximate uncertainty quantification methodologies.
The standard L96 equations are implemented with an additional forcing term with time dependence.

#### Lorenz 96 equations

The standard single-scale L96 equations are implemented.
The Lorenz 96 system \cite{lorenz1996predictability} is given by 
```math
\frac{d x_i}{d t} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,
```
with ```i``` indicating the index of the given longitude. The number of longitudes is given by ```N```.
The boundary conditions are given by
```math
x_{-1} = x_{N-1}, x_0 = x_N, x_{N+1} = x_1.
```
The time scaling is such that the characteristic time is 5 days. 
For very small values of ``F``, the solutions $X_i$ decay to $F$ after the initial transient feature.
For moderate values of ``F``, the solutions are periodic, and for larger values of ``F``, the system is chaotic.
The solution variance is a function of the forcing magnitude.
Variations in the base state as a function of time can be imposed through a time-dependent forcing term ``F(t)``.

A temporal forcing term is defined
```math
F = F_s + A \sin(\omega t),
```
with steady-state forcing ``F_s``, transient forcing amplitude ``A``, and transient forcing frequency ``\omega``.

The L96 dynamics are solved with RK4 integration.
The system is solved over time horizon ```0``` to `tend` at fixed time step `dt`.


#### Structure

The main code is located in `Lorenz_example.jl` which provides the functionality to run the L96 dynamical system, extract time-averaged statistics from the L96 states, and use the time-average statistics for optimization and uncertainty quantification.

The L96 system is solved in `GModel.jl` according to the time integration settings specified in `LSettings` and the L96 parameters specified in `LParams`.
The types of statistics to be collected are detailed in `GModel.jl`.


#### Lorenz dynamics inputs

### Dynamics settings
The use of the transient forcing term is with the flag, `dynamics`. Stationary forcing is `dynamics=1` and transient forcing is used with `dynamics=2`.
The default parameters are specified in `Lorenz_example.jl` and can be modified as necessary.

### Inverse problem settings
The states are integrated over time `Ts_days` to construct the time averaged statistics for use by the optimization.
The specification of the statistics to be gathered from the states are provided by `stats_type`.
```julia
N_ens = 20 # number of ensemble members
N_iter = 5 # number of EKI iterations
```


#### Setting up the Inverse Problem
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
We use wide, normal priors without constraints

```julia
prior_means = [F_true+1.0, A_true+0.5]
prior_stds = [2.0, 0.5*A_true]
d1 = Parameterized(Normal(prior_means[1], prior_stds[1]))
d2 = Parameterized(Normal(prior_means[2], prior_stds[2]))
prior_distns = [d1, d2]
c1 = no_constraint()
c2 = no_constraint()
constraints = [[c1], [c2]]
prior_names = param_names
priors = ParameterDistribution(prior_distns, constraints, prior_names)
```

### Observational Noise
The observational noise can be generated using the L96 system or prescribed, as specified by `var_prescribe`. 

## `var_prescribe==true`
The observational noise is constructed by generating independent instantiations of the L96 statistics of interest at the true parameters for different initial conditions.
The empirical covariance matrix is constructed.

## `var_prescribe==false`
The observational noise is prescribed as a Gaussian distribution with prescribed mean and variance.

#### Running the Example
The L96 parameter estimation can be run using `julia --project Lorenz_example.jl`


### Solution and Output
The output will provide the estimated parameters 

## Printed output
```julia
# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)
println("\nEKI results:")
println(mean(get_u_final(ekiobj), dims=2))
```

## Saved output
The parameters and forward model outputs will be saved in `parameter_storage.jld2` and `data_storage.jld2`, respectively.

## Plots
A scatter plot of the parameter estimates compared to the true parameters will be provided.
