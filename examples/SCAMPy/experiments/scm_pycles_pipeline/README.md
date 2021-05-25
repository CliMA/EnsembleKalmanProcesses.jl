## Training the SCAMPy EDMF with PyCLES data: An example

Author: Ignacio Lopez-Gomez

Last updated : May 2021

To perform simulations, first we need to clone the SCAMPy repo to this directory,

  >> git clone https://github.com/CliMA/SCAMPy.git

And of course, compile it. Here are the steps to do so in the Caltech HPC,

  >> module purge 

  >> module load hdf5/1.10.1

  >> module load netcdf-c/4.6.1

  >> module load netcdf-cxx/4.3.0

  >> module load netcdf-fortran/4.4.4

  >> module load openmpi/1.10.7

  >> CC=mpicc python setup.py build_ext --inplace

Then, we need to copy the following script to the SCAMPy folder,

  >> cp call_BOMEX.sh SCAMPy/

This is the script that modifies parameters in the SCAMPy input file. Note that in this directory (*scm_pycles_pipeline*), there is a SCAMPy output directory. The files within this output directory are used by the code to interpolate the LES data to the SCM resolution and to send the modified parameters to SCAMPy. 

The parameters that we seek to optimize should be modified to a value of 0.01 initially (already done here) within the *paramlist* of this folder. Also note that the Bomex.in file has uuid ending in *51515*. This is also used to modify the uuid by the script, and should not be modified. The resolution at which the simulation will be run, etc, is specified in these input files and can be modified based on the needs of the user.

After this, we should be ready to go. You can run the simulation by writing

  >> sbatch calibrate_script

After the simulations are done, the results will be stored in a directory named *results_...*
Some output of interest is included both in numpy and jld formats. 
Examples of some of the results may be found in the example output directory *results_eki_p2_e20_i10_d7*, which includes several plots of the evolution of the parameters and the loss function.
