## Training the TurbulenceConvection.jl EDMF with PyCLES data: An example

Authors: Yair Cohen, Haakon Ludvig Langeland Ervik, Ignacio Lopez-Gomez, Zhaoyi Shen

Last updated : July 2021

This example script details the calibration of parameters for a single-column EDMF model implemented in `TurbulenceConvection.jl` using PyCLES data as a reference. The script also enable calibration in the perfect-model setting.

To perform simulations, first we need to clone this repo and the `TurbulenceConvection.jl` repo from GitHub.

  >> git clone https://github.com/ilopezgp/EnsembleKalmanProcesses.jl

  >> git clone https://github.com/CliMA/TurbulenceConvection.jl 

Before calibration, we need to compile the project. Note that each example in EnsembleKalmanProcesses.jl is its own project, so we will have to perform compilation within each example in the `examples` directory. In this case, stay within `.../EnsembleKalmanProcesses.jl/examples/SCAMPy`. For the time being neither `EnsembleKalmanProcesses.jl` nor `TurbulenceConvection.jl` are published packages, so the dependencies are added manually by

>> julia --project

>> julia> ]

>> pkg> dev path/to/TurbulenceConvection.jl

>> pkg> dev path/to/EnsembleKalmanProcesses.jl

>> pkg> instantiate

Then, navigate into the `scm_pycles_pipeline` directory (i.e. this example), and open `calibrate.jl`.

A number of variables are exposed that allow you to select where to fetch input data, and where to store the output. For input data, you should:

- update `les_names`, `les_suffixes` and `les_root` to use different LES reference data
    - By default, Ignacio's bomex data is used.
- update `scm_names` and `scm_data_root` to point to another SCM reference folder. Note that `scm_data_root` should point to a directory containing the folder `Output.<scm_name>.00000` where `<scm_name>` is an element in `scm_names`. This folder is read (but not modified) when running forward simulations and updating parameters. The LES data will be interpolated to the resolution defined by the `.nc` datafile in the Output folder.
    - By default it points to `Output.Bomex.00000` in this directory.

For output data, you should

- update `outdir_root` to point to a directory where the output folder will be placed. By default, it is this directory.
    - Data that is stored here include various statistics and diagnostics used by the EKI method, and  (not by default) `.nc` datafiles (timeseries) for each ensemble simulation. 


If you are on the Caltech Central Cluster, you run the project by adding it to the schedule by,

  >> sbatch calibrate_script

otherwise run locally by e.g.

>> sh calibrate_script

After the simulations are done, the results will be stored in a directory named *results_...*
Output of interest is stored in jld2 format. 
Examples of some of the results may be found in the example output directory *results_eki_p2_e20_i10_d7*, which includes several plots of the evolution of the parameters and the loss function.
See the `.../EnsembleKalmanProcesses.jl/examples/SCAMPy/plot` folder to create some simple plots based on the output data.
