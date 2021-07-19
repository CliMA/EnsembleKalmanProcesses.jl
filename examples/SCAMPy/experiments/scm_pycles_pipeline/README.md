## Training the SCAMPy EDMF with PyCLES data: An example

Author: Ignacio Lopez-Gomez

Last updated : June 2021

To perform simulations, first we need to clone the SCAMPy repo from GitHub. SCAMPy can be cloned to any directory because its path is made known to this example by the `scampy_dir` variable (see below). By default `scampy_dir` will refer to the `scm_pycles_pipeline` directory (i.e. this example).

  >> git clone https://github.com/CliMA/SCAMPy.git

SCAMPy must be compiled. Navigate into the `SCAMPy` folder, and for the Caltech HPC, do

  >> module purge 

  >> module load hdf5/1.10.1 netcdf-c/4.6.1 netcdf-cxx/4.3.0 netcdf-fortran/4.4.4 openmpi/1.10.7

  >> CC=mpicc python setup.py build_ext --inplace

Then, navigate into the `scm_pycles_pipeline` directory (i.e. this example), and open `calibrate.jl`.

A number of variables are exposed that allow you to select where to fetch input data, and where to store the output. For input data, you should:

- update `scampy_dir` to point to the directory of your local `SCAMPy` repo.
    - By default, it points to `./SCAMPy`, which you can leave unchanged `SCAMPy` was cloned into this example directory
- update `les_names`, `les_suffixes` and `les_root` to use different LES reference data
    - By default, Ignacio's bomex data is used.
- update `scm_names` and `scm_data_root` to point to another SCM reference folder. Note that `scm_data_root` should point to a directory containing the folder `Output.<scm_name>.00000` where `<scm_name>` is an element in `scm_names`. This folder is read (but not modified) when running forward simulations and updating parameters. The LES data will be interpolated to the resolution defined by the `.nc` datafile in the Output folder.
    - By default it points to `Output.Bomex.00000` in this directory.

For output data, you should

- update `outdir_root` to point to a directory where the output folder will be placed. By default, it is this directory.
    - Data that is stored here include `.nc` datafiles (timeseries) for each ensemble simulation, and various statistics and diagnostics used by the EKI method.

Before calibration, we need to compile the Julia project. Note that each example in EnsembleKalmanProcesses.jl is its own project, so we will have to perform compilation within each example in the `examples` directory. In this case, stay within `.../EnsembleKalmanProcesses.jl/examples/SCAMPy`. For the time being `EnsembleKalmanProcesses.jl` is not a published package, so in order to precompile, you will need to add the dependency to your local package by

>> julia --project

>> ]

>> rm EnsembleKalmanProcesses

>> dev path/to/EnsembleKalmanProcesses.jl

>> instantiate

where the second line indicates entering the [Pkg REPL](https://docs.julialang.org/en/v1/stdlib/Pkg/#Pkg).

You can run the simulation by writing

  >> sbatch calibrate_script

After the simulations are done, the results will be stored in a directory named *results_...*
Some output of interest is included both in numpy and jld formats. 
Examples of some of the results may be found in the example output directory *results_eki_p2_e20_i10_d7*, which includes several plots of the evolution of the parameters and the loss function.
