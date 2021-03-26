#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ces_init"   # job name

#julia package management
module load julia/1.5.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

julia --project init_calibration.jl
echo 'Ensemble initialized for calibration.'
mv clima_param_defs* ../../../ClimateMachine.jl/test/Atmos/EDMF/
