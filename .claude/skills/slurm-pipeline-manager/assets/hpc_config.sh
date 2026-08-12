#!/usr/bin/env bash
# =============================================================================
# hpc_config.sh  —  Single source of truth for all SLURM / HPC settings.
# Edit this file before every run. Source it from every .sbatch and .sh script.
# =============================================================================

# ---------------------------------------------------------------------------
# Julia / environment
# ---------------------------------------------------------------------------
JULIA_MODULE="julia/1.12.2"     # module name on your cluster (julia/X.Y.Z)
JULIA_PROJECT="."               # --project arg; "." uses Project.toml in submit dir

# ---------------------------------------------------------------------------
# SLURM account / partition
# ---------------------------------------------------------------------------
ACCOUNT="TODO_your_account"     # -A flag (required on most clusters)
PARTITION=""                    # leave empty to use cluster default

# ---------------------------------------------------------------------------
# Output root and run date
# ---------------------------------------------------------------------------
# RUN_DATE pins the output directory for ALL jobs in this tree.
# Set it ONCE before submitting and do not change it mid-run.
# Tip: use today's date when starting a new run; keep it fixed for reruns.
RUN_DATE="__RUN_DATE__"         # e.g. 2026-01-15  (filled in by slurm-pipeline-manager)
OUTPUT_ROOT="output"            # relative to the submit directory
# Full output dir for this run: $OUTPUT_ROOT/$RUN_DATE/

# ---------------------------------------------------------------------------
# Pipeline scripts  (relative paths from the example directory)
# ---------------------------------------------------------------------------
SETUP_SCRIPT="__SETUP_SCRIPT__"           # e.g. initialize_EKP.jl
FORWARD_SCRIPT="__FORWARD_SCRIPT__"       # e.g. run_computer_model.jl  (receives iteration member)
UPDATE_SCRIPT="__UPDATE_SCRIPT__"         # e.g. update_EKP.jl          (receives iteration)
POSTPROCESS_SCRIPT="__POSTPROCESS_SCRIPT__"  # e.g. postprocess.jl  (leave empty if none)

# ---------------------------------------------------------------------------
# EKP budget
# ---------------------------------------------------------------------------
N_ENSEMBLE=__N_ENSEMBLE__       # number of ensemble members (sets --array=1-N)
N_ITERATIONS=__N_ITERATIONS__   # number of EKP iterations

# ---------------------------------------------------------------------------
# Per-stage SLURM resources
# ---------------------------------------------------------------------------
# Precompile
PRECOMPILE_TIME="00:30:00"
PRECOMPILE_MEM="8G"
PRECOMPILE_CPUS=4

# Setup / init
SETUP_TIME="00:30:00"
SETUP_MEM="8G"
SETUP_CPUS=2

# Forward map (per member)
FWD_TIME="02:00:00"
FWD_MEM="16G"
FWD_CPUS=4
FWD_ARRAY_THROTTLE="%50"        # max simultaneous tasks, e.g. %50. Empty = no limit.

# Update ensemble
UPDATE_TIME="00:30:00"
UPDATE_MEM="8G"
UPDATE_CPUS=2

# Postprocessing
POST_TIME="01:00:00"
POST_MEM="8G"
POST_CPUS=2
