#!/usr/bin/env bash
# =============================================================================
# forward_map.sbatch  —  Per-member forward model evaluation (array job).
#
# Submitted by run_pipeline.sh as an array job (--array=1-N_ENSEMBLE).
# The SLURM array task ID is the ensemble member index (1-based).
# The EKP iteration index is passed via --export=ALL,ITERATION=<i>.
#
# Dependencies (set by run_pipeline.sh):
#   First iteration : afterok:<setup_jid>
#   Later iterations: afterok:<update_jid_for_previous_iteration>
# =============================================================================
#SBATCH --job-name=ekp_fwd
#SBATCH --output=__OUTPUT_ROOT__/__RUN_DATE__/slurm/fwd_%A_%a_iter%x.out
#SBATCH --error=__OUTPUT_ROOT__/__RUN_DATE__/slurm/fwd_%A_%a_iter%x.err
#SBATCH --array=1-__N_ENSEMBLE____FWD_ARRAY_THROTTLE__
#SBATCH --time=__FWD_TIME__
#SBATCH --mem=__FWD_MEM__
#SBATCH --cpus-per-task=__FWD_CPUS__
#SBATCH --ntasks=1

set -euo pipefail

source "$(dirname "$0")/hpc_config.sh"

module load "${JULIA_MODULE}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# IMPORTANT: precompile has already run; suppress re-precompilation.
export JULIA_PKG_PRECOMPILE_AUTO=0

cd "${SLURM_SUBMIT_DIR}"

# ITERATION is injected via --export=ALL,ITERATION=<i> at submit time.
: "${ITERATION:?ITERATION env var not set — submit via run_pipeline.sh}"

echo "Running forward map: iteration=${ITERATION}, member=${SLURM_ARRAY_TASK_ID}"

julia --project="${JULIA_PROJECT}" "${FORWARD_SCRIPT}" \
    "${OUTPUT_ROOT}/${RUN_DATE}" \
    "${ITERATION}" \
    "${SLURM_ARRAY_TASK_ID}"

echo "Forward map done: iteration=${ITERATION}, member=${SLURM_ARRAY_TASK_ID}"
