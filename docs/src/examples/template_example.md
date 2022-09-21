# Template example

We provide the following template for how the tools may be applied.

For small examples typically have 2 files.

- `DynamicalModel.jl` Contains the dynamical model ``\Psi`` and the observation map ``\mathcal{H}``. The inputs should be the so-called free parameters (in the constrained/physical space that is the input domain of the dynamical model) we are interested in learning, and the output should be the measured data.
- The example script which contains the inverse problem setup and solve

## The structure of the example script

### Create the data and the setting for the model
1. Set up the forward model.
2. Construct/load the truth data. 

### Set up the inverse problem
3. Define the prior distributions, and generate an initial ensemble.
4. Initialize the `process` tool you would like to use (we recommend you begin with `Inversion()`). 
5. initialize the `EnsembleKalmanProcess` object

### Solve the inverse problem, in a loop

7. Obtain the current parameter ensemble
8. Transform them from the unbounded computational space to the physical space
9. call the forward model on the ensemble of parameters, producing an ensemble of measured data
10. call the `update_ensemble!` function to generate a new parameter ensemble based on the new data

### Get the solution
1. Obtain the final parameter ensemble, compute desired statistics here.
2. Transform the final ensemble into the physical space for use in prediction studies with the forward model.
