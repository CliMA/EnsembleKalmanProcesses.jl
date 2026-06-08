#!/usr/bin/env bash
# =============================================================================
# postprocess.sbatch  —  Postprocessing / plotting after all EKP iterations.
#
# Submitted by run_pipeline.sh after the final update_ensemble job completes.
# Can also be run standalone via run_postprocess.sh.
#
# Dependency (when in pipeline): afterok:<final_update_jid>
# =============================================================================
#SBATCH --job-name=ekp_post
#SBATCH --output=__OUTPUT_ROOT__/__RUN_DATE__/slurm/post_%j.out
#SBATCH --error=__OUTPUT_ROOT__/__RUN_DATE__/slurm/post_%j.err
#SBATCH --time=__POST_TIME__
#SBATCH --mem=__POST_MEM__
#SBATCH --cpus-per-task=__POST_CPUS__
#SBATCH --ntasks=1

set -euo pipefail

source "$(dirname "$0")/hpc_config.sh"

module load "${JULIA_MODULE}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# IMPORTANT: precompile has already run; suppress re-precompilation.
export JULIA_PKG_PRECOMPILE_AUTO=0

cd "${SLURM_SUBMIT_DIR}"

if [[ -z "${POSTPROCESS_SCRIPT}" ]]; then
    echo "POSTPROCESS_SCRIPT is empty — nothing to run. Set it in hpc_config.sh."
    exit 0
fi

echo "Running postprocessing: ${POSTPROCESS_SCRIPT}"

julia --project="${JULIA_PROJECT}" "${POSTPROCESS_SCRIPT}" \
    "${OUTPUT_ROOT}/${RUN_DATE}"

echo "Postprocessing done."
