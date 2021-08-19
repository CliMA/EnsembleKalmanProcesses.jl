# TurbulenceConvection.jl example


## Overview

`TurbulenceConvection.jl` is a a single-column atmospheric model in Julia. The model implements the extended eddy-diffusivity mass-flux closure for unified representation of subgrid-scale turbulence and convection. In this example, we demonstrate the calibration of parameters for dynamic entrainment and detrainment, which represent key processes for subgrid-scale mixing. Finite time-averaged statistics of grid-mean quantities are obtained from high-resolution Large Eddy Simulations, and are used as reference data in the calibration process.


## Extended eddy-diffusivity mass-flux equations

The extended eddy-diffusivity mass-flux closure is implemented as described in [Lopez-Gomez et al. 2020]() and [Cohen et al. 2020]().

**Equations**


## Structure

The main configuration of the calibration is located in `calibrate.jl` which defines the parameters to be calibrated and their prior distributions,
```julia
params = Dict(
    # entrainment parameters
    "entrainment_factor"        => [bounded(0.0, 1.5*0.33)],
    "detrainment_factor"        => [bounded(0.0, 1.5*0.31)],
)
```
as well as the names of the variables used in the loss function and location of reference data used to compute finite time-averaged statistics, which is stored in the `ReferenceModel` construct.

Various flags allow us to easily decide whether to `perform_PCA` on the reference statistics, `normalize` the reference and forward model statistics, use LES or SCM data as reference (`model_type`), and restrict which type of output data is stored (`save_eki_data`, `save_ensemble_data`). 

For the inversion step, we also specify the number of ensemble members `N_ens` and the number of iterations of the inversion method `N_iter`.

Several methods and structures that enamble computation of the forward model, retrieval of time-averaged statistics or performing PCA is stored in the `helper_funcs.jl` file in the `src` folder of the `SCM` example directory.


## Running the Example
The parameter estimation example can be run locally using `sh calibrate_script`.

If you're on the Caltech Central Cluster, `sbatch calibrate_script` can be invoked to allocate the required resources. Note that the `calibrate_script` file allocates the necessary simulation time and number of tasks. This should be adjusted if different setups are considered.


## Solution and Output
The output will provide the estimated parameters.

### Saved output
All output will be saved in a folder starting with `results_eki_`.
Data related to the Ensemble Kalman Inversion will be stored in `ekp.jld2`. 

### Plots
Plots of parameter and error evolution will be provided in the output folder.
