#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "fms_ekp"   # job name
#SBATCH -o output/slurm/array-%A_%a.out
#SBATCH -e output/slurm/array-%A_%a.err

module load julia/1.8.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1
iteration_=${1?Error: no iteration given}

run_num=${SLURM_ARRAY_TASK_ID}

julia output/output_$run_num/input_file

