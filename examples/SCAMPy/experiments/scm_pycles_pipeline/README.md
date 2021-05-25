To perform the simulation, first we need to clone the SCAMPy repo to this directory,

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

After this, we should be ready to go. You can run the simulation by writing

  >> sbatch calibrate_script
