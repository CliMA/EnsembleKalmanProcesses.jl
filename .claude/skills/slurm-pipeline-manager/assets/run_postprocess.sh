#!/usr/bin/env bash
# =============================================================================
# run_postprocess.sh  —  Submit ONLY the postprocessing job.
#
# Use this to re-run postprocessing on an existing completed output tree,
# without re-running the full calibration. Reads RUN_DATE from hpc_config.sh.
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/hpc_config.sh"

if [[ -z "${POSTPROCESS_SCRIPT:-}" ]]; then
    echo "POSTPROCESS_SCRIPT is not set in hpc_config.sh — nothing to do."
    exit 0
fi

if [[ ! -d "${OUTPUT_ROOT}/${RUN_DATE}" ]]; then
    echo "Output directory ${OUTPUT_ROOT}/${RUN_DATE} does not exist."
    echo "Check RUN_DATE in hpc_config.sh."
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/${RUN_DATE}/slurm"

SBATCH_ARGS=(-A "${ACCOUNT}")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=(--partition "${PARTITION}")

JID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${DIR}/postprocess.sbatch")
echo "Submitted postprocess job: ${JID}"
echo "Watch with: squeue -j ${JID}"
echo "Logs: ${OUTPUT_ROOT}/${RUN_DATE}/slurm/"
