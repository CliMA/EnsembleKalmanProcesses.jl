# Lorenz 96 example

### Overview

The Lorenz 96 (hereafter L96) example is a toy-problem for the application of the EnsembleKalmanPRocesses.jl optimization and approximate uncertainty quantification methodologies.

### Lorenz 96 equations

The standard single-scale L96 equations are implemented.
The Lorenz 96 system \cite{lorenz1996predictability} is given by 
```math
\frac{d x_i}{d t} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F.
```
The boundary conditions are given by
```math
x_{-1} &= x_{N-1}
x_0 &= x_N, & x_{N+1} &= x_1.
```
The time scaling is such that the characteristic time is 5 days. For very small values of ``F``, the solutions $X_i$ decay to $F$ after the initial transient feature.
For moderate values of ``F``, the solutions are periodic, and for larger values of ``F``, the system is chaotic.
The solution variance is a function of the forcing magnitude.
Variations in the base state as a function of time can be imposed through a time-dependent forcing term ``F(t)``.

### Prerequisites

### Structure

### Setting up the Inverse Problem

#### Priors

#### Observational Noise

### Running the Example

### Solution and Output


