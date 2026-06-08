#!/usr/bin/env bash
# =============================================================================
# setup.sbatch  —  EKP setup: build prior, initial ensemble, save eki.jld2,
# write iteration-0 member parameter files.
#
# Depends on: precompile (run manually via run_precompile.sh first).
# This job is submitted by run_pipeline.sh with --dependency=afterok:<setup_jid>
# for the first forward-map array job.
# =============================================================================
#SBATCH --job-name=ekp_setup
#SBATCH --output=__OUTPUT_ROOT__/__RUN_DATE__/slurm/setup_%j.out
#SBATCH --error=__OUTPUT_ROOT__/__RUN_DATE__/slurm/setup_%j.err
#SBATCH --time=__SETUP_TIME__
#SBATCH --mem=__SETUP_MEM__
#SBATCH --cpus-per-task=__SETUP_CPUS__
#SBATCH --ntasks=1

set -euo pipefail

source "$(dirname "$0")/hpc_config.sh"

module load "${JULIA_MODULE}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# IMPORTANT: precompile has already run; suppress re-precompilation.
export JULIA_PKG_PRECOMPILE_AUTO=0

mkdir -p "${OUTPUT_ROOT}/${RUN_DATE}/slurm"
cd "${SLURM_SUBMIT_DIR}"

julia --project="${JULIA_PROJECT}" "${SETUP_SCRIPT}" \
    "${OUTPUT_ROOT}/${RUN_DATE}"

echo "Setup complete. Output in ${OUTPUT_ROOT}/${RUN_DATE}/"
