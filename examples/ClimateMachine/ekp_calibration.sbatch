#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "ekp_call"   # job name

# Size of the ensemble
n=10
# Number of EK iterations
n_it=5

module purge
module load julia/1.5.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# run instantiate/precompile serial
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'

# First call to calibrate.jl will create the ensemble files from the priors
id_init_ens=$(sbatch --parsable -A esm ekp_init_calibration.sbatch)
for it in $(seq 1 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "1" ]; then
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_init_ens --array=1-$n ekp_single_cm_run.sbatch $it)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ek_upd --array=1-$n ekp_single_cm_run.sbatch $it)
fi
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes -A esm --dependency=afterok:$id_ens_array --export=n=$n ekp_cont_calibration.sbatch $it)
done

