#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Submit the full EKP dependency tree.
#
# Builds the job graph:
#   setup
#     └─afterok─► fwd_iter_0 [array 1..N] ─afterok─► update_iter_0
#     └─afterok─► fwd_iter_1 [array 1..N] ─afterok─► update_iter_1
#         ...
#     └─afterok─► fwd_iter_K-1 [array 1..N] ─afterok─► update_iter_K-1
#                                                        └─afterok─► postprocess
#
# All run-stage jobs set JULIA_PKG_PRECOMPILE_AUTO=0.
# Run run_precompile.sh first and wait for it to finish.
#
# Usage:
#   bash run_pipeline.sh          # submit for real
#   bash run_pipeline.sh --dry-run  # print sbatch commands without submitting
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/hpc_config.sh"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${OUTPUT_ROOT}/${RUN_DATE}/slurm"

# ---------------------------------------------------------------------------
# Helper: submit a job and return its job ID (or a placeholder in dry-run).
# Usage: submit_job <label> [extra sbatch args...] -- <sbatch_file>
# ---------------------------------------------------------------------------
_JID_COUNTER=1000
submit_job() {
    local label="$1"; shift
    local -a extra_args=()
    while [[ "$1" != "--" ]]; do
        extra_args+=("$1"); shift
    done
    shift  # consume "--"
    local sbatch_file="$1"

    local -a base_args=(-A "${ACCOUNT}")
    [[ -n "${PARTITION:-}" ]] && base_args+=(--partition "${PARTITION}")

    if "${DRY_RUN}"; then
        _JID_COUNTER=$(( _JID_COUNTER + 1 ))
        # Print readable summary to stderr so it doesn't pollute the captured JID
        echo "[DRY RUN] ${label}:" \
             "sbatch --parsable ${base_args[*]} ${extra_args[*]} ${sbatch_file}" \
             "-> JID=${_JID_COUNTER}" >&2
        echo "${_JID_COUNTER}"   # only the numeric ID goes to stdout / gets captured
    else
        sbatch --parsable "${base_args[@]}" "${extra_args[@]}" "${sbatch_file}"
    fi
}

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
SETUP_JID=$(submit_job "setup" \
    --job-name="ekp_setup" \
    -- "${DIR}/setup.sbatch")
echo "Setup job ID: ${SETUP_JID}"

PREV_JID="${SETUP_JID}"

# ---------------------------------------------------------------------------
# 2. Iteration loop: forward map (array) + update_ensemble
# ---------------------------------------------------------------------------
for (( i=0; i<N_ITERATIONS; i++ )); do

    # Array throttle (append to --array value)
    ARRAY_SPEC="1-${N_ENSEMBLE}${FWD_ARRAY_THROTTLE}"

    FWD_JID=$(submit_job "fwd iter ${i}" \
        --job-name="ekp_fwd_iter${i}" \
        --array="${ARRAY_SPEC}" \
        --dependency="afterok:${PREV_JID}" \
        --kill-on-invalid-dep=yes \
        --export="ALL,ITERATION=${i}" \
        -- "${DIR}/forward_map.sbatch")
    echo "  Forward map iter ${i}: JID=${FWD_JID}"

    UPDATE_JID=$(submit_job "update iter ${i}" \
        --job-name="ekp_update_iter${i}" \
        --dependency="afterok:${FWD_JID}" \
        --kill-on-invalid-dep=yes \
        --export="ALL,ITERATION=${i}" \
        -- "${DIR}/update_ensemble.sbatch")
    echo "  Update       iter ${i}: JID=${UPDATE_JID}"

    PREV_JID="${UPDATE_JID}"
done

# ---------------------------------------------------------------------------
# 3. Postprocessing (runs after final update, even if empty)
# ---------------------------------------------------------------------------
POST_JID=$(submit_job "postprocess" \
    --job-name="ekp_post" \
    --dependency="afterok:${PREV_JID}" \
    --kill-on-invalid-dep=yes \
    -- "${DIR}/postprocess.sbatch")
echo "Postprocess job ID: ${POST_JID}"

echo ""
if "${DRY_RUN}"; then
    echo "DRY RUN complete. No jobs were submitted."
else
    echo "Pipeline submitted. Monitor with: squeue -u \$USER"
    echo "Logs: ${OUTPUT_ROOT}/${RUN_DATE}/slurm/"
fi
