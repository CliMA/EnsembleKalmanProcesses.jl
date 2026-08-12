#!/usr/bin/env bash
# =============================================================================
# run_precompile.sh  —  Submit ONLY the precompile job.
#
# Run this once before your first pipeline submission, and any time you update
# Julia packages (Project.toml / Manifest.toml).
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/hpc_config.sh"

mkdir -p "${OUTPUT_ROOT}/${RUN_DATE}/slurm"

SBATCH_ARGS=(-A "${ACCOUNT}")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=(--partition "${PARTITION}")

JID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${DIR}/precompile.sbatch")
echo "Submitted precompile job: ${JID}"
echo "Watch with: squeue -j ${JID}"
echo "When it completes, run: bash run_pipeline.sh"
