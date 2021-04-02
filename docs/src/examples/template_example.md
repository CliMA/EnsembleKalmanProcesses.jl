# Template example

We provide the following template for how the tools may be applied.

For small examples typically have 2 files.

- `GModel.jl` Contains the forward map. The inputs should be the so-called free parameters we are interested in learning, and the output should be the measured data
- The example script which contains the inverse problem setup and solve

## The structure of the example script
First we create the data and the setting for the model
1. Set up the forward model.
2. Construct/load the truth data. Store this data conveniently in the `Observations.Obs` object

Then we set up the inverse problem
3. Define the prior distributions. Use the `ParameterDistribution` object
4. Decide on which `process` tool you would like to use (we recommend you begin with `Invesion()`). Then initialize this with the relevant constructor
5. initialize the `EnsembleKalmanProcess` object

Then we solve the inverse problem, in a loop perform the following for as many iterations as required:
7. Obtain the current parameter ensemble
8. Transform them from the unbounded computational space to the physical space
9. call the forward map on the ensemble of parameters, producing an ensemble of measured data
10. call the `update_ensemble!` function to generate a new parameter ensemble based on the new data

One can then obtain the solution, dependent on the `process` type.
