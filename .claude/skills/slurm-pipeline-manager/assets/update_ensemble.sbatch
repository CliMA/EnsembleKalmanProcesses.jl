#!/usr/bin/env bash
# =============================================================================
# update_ensemble.sbatch  —  Collect forward-map outputs and call update_ensemble!
#
# Submitted by run_pipeline.sh after the forward-map array completes.
# The EKP iteration is passed via --export=ALL,ITERATION=<i>.
#
# Dependency: afterok:<forward_array_jid_for_this_iteration>
# =============================================================================
#SBATCH --job-name=ekp_update
#SBATCH --output=__OUTPUT_ROOT__/__RUN_DATE__/slurm/update_%j_iter%x.out
#SBATCH --error=__OUTPUT_ROOT__/__RUN_DATE__/slurm/update_%j_iter%x.err
#SBATCH --time=__UPDATE_TIME__
#SBATCH --mem=__UPDATE_MEM__
#SBATCH --cpus-per-task=__UPDATE_CPUS__
#SBATCH --ntasks=1

set -euo pipefail

source "$(dirname "$0")/hpc_config.sh"

module load "${JULIA_MODULE}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# IMPORTANT: precompile has already run; suppress re-precompilation.
export JULIA_PKG_PRECOMPILE_AUTO=0

cd "${SLURM_SUBMIT_DIR}"

: "${ITERATION:?ITERATION env var not set — submit via run_pipeline.sh}"

echo "Running update_ensemble!: iteration=${ITERATION}"

julia --project="${JULIA_PROJECT}" "${UPDATE_SCRIPT}" \
    "${OUTPUT_ROOT}/${RUN_DATE}" \
    "${ITERATION}"

echo "Update done: iteration=${ITERATION}"
