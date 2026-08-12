#!/usr/bin/env bash
# =============================================================================
# precompile.sbatch  —  One-time Julia package precompile.
#
# Run this BEFORE submitting run_pipeline.sh. All other stages set
# JULIA_PKG_PRECOMPILE_AUTO=0 and rely on the warm cache this job produces.
# =============================================================================
#SBATCH --job-name=precompile
#SBATCH --output=__OUTPUT_ROOT__/__RUN_DATE__/slurm/precompile_%j.out
#SBATCH --error=__OUTPUT_ROOT__/__RUN_DATE__/slurm/precompile_%j.err
#SBATCH --time=__PRECOMPILE_TIME__
#SBATCH --mem=__PRECOMPILE_MEM__
#SBATCH --cpus-per-task=__PRECOMPILE_CPUS__
#SBATCH --ntasks=1

set -euo pipefail

source "$(dirname "$0")/hpc_config.sh"

module load "${JULIA_MODULE}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTPUT_ROOT}/${RUN_DATE}/slurm"
cd "${SLURM_SUBMIT_DIR}"

# NOTE: This is the ONLY job that precompiles.
# All run-stage jobs set JULIA_PKG_PRECOMPILE_AUTO=0.
julia --project="${JULIA_PROJECT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Precompile complete."
