---
name: slurm-pipeline-manager
description: >
  Scaffold and maintain a SLURM/HPC job-dependency tree for an
  EnsembleKalmanProcesses.jl (EKP) calibration pipeline. Invoke this skill
  whenever the user wants to run an EKP calibration on a cluster, HPC system,
  or job scheduler — even if they don't say "SLURM" explicitly. Trigger phrases
  include: "get my calibration running on the cluster", "parallelize the ensemble
  over HPC", "set up sbatch for this inversion", "submit this EKP run to
  slurm/HPC", "make this pipeline work on our HPC", "add slurm support", "run
  ensemble members in parallel on the cluster", and maintenance phrasings such as
  "update / fix / regenerate my slurm pipeline", "change the time limit on my
  HPC jobs", "add a postprocessing job to the pipeline". Also trigger when the
  user describes a multi-stage EKP calibration they want to run on a cluster
  without naming SLURM at all. Use this skill proactively whenever HPC, cluster,
  job scheduler, sbatch, or ensemble parallelism is mentioned in a Julia
  calibration context.
---

# slurm-pipeline-manager

This skill reads a user's decoupled EKP Julia pipeline, verifies it is
structurally correct, and generates a complete SLURM job-dependency tree into a
`slurm-variant/` subdirectory inside the example directory. Keeping generated
files in `slurm-variant/` keeps the Julia source uncluttered and makes it easy
to version-control or delete the HPC scaffolding independently.

The generated tree uses a one-shot precompile job followed by a
dependency-chained `setup → forward_map[array] → update_ensemble` loop across
iterations, ending in postprocessing — so ensemble members run in parallel and
every stage pays precompile cost only once.

It also supports **maintenance mode**: if the pipeline already has a
`slurm-variant/` subdirectory with generated HPC scripts, re-invoking the skill
updates them in place without clobbering the user's manifest settings.

---

## How the generated pipeline is structured

```
precompile ─(manual, run first)

setup ──afterok──► fwd_iter_0[array 1..N] ──afterok──► update_iter_0
      (afterok chain continues for each iteration)
      ──afterok──► fwd_iter_K-1[array 1..N] ──afterok──► update_iter_K-1
                                                         ──afterok──► postprocess
```

Key design choices (understand these so you can explain them to the user):
- **One precompile job only.** Every run-stage job sets
  `JULIA_PKG_PRECOMPILE_AUTO=0`. Without this, 60 concurrent array tasks would
  each attempt to precompile — thrashing the shared Julia depot and wasting time.
- **Forward map as a SLURM array.** `--array=1-$N_ENSEMBLE` gives one task per
  ensemble member; the iteration index travels via `--export=ALL,ITERATION=$i`.
- **`afterok` + `--kill-on-invalid-dep=yes` everywhere.** If one stage fails the
  whole tree is killed, not left stuck in the queue. EKP itself errors on < 2
  successful members (commit 8f2d3fa), so partial-run continuation is not useful.
- **Date-stamped output root.** `RUN_DATE` in `hpc_config.sh` is the outermost
  directory, so independent runs stay separated.

---

## Workflow

### Mode 0 — Detect generate vs. maintenance

Check whether `<example_dir>/slurm-variant/hpc_config.sh` already exists. If so,
enter **maintenance mode**: read the existing `hpc_config.sh` as the source of
truth, understand the user's requested change, and edit only what is necessary
inside `slurm-variant/` — do **not** overwrite `hpc_config.sh` with fresh
template defaults. Common maintenance requests: resource changes, adding/removing
a stage, updating script filenames after the user refactored their Julia,
re-generating the sbatch headers after an N_ensemble change.

If `slurm-variant/` does not yet exist, proceed with generate mode below.

---

### Step 1 — Read the example and map the pipeline stages

Read every `.jl` file in the named example directory. Identify which file plays
each role and note the key function calls and file arguments:

| Role | What to look for |
|---|---|
| **Setup/init** | builds `prior`, `EnsembleKalmanProcess`, saves `eki`, `param_dict`, `prior` to JLD2, calls `save_parameter_ensemble` for iteration 0 |
| **Forward map** | reads `parameters.toml` from a member dir, runs the model, writes `output.jld2`; must accept `iteration` and `member` as args |
| **Update** | loads `eki.jld2`, loops members collecting `G_ens`, calls `update_ensemble!`, saves next iteration TOMLs and updated `eki.jld2` |
| **Postprocess** | plotting / analysis — no EKP update; may be absent |
| **Data generation** | creates truth + noise; usually a one-off, may be merged into setup |

Record `SETUP_SCRIPT`, `FORWARD_SCRIPT`, `UPDATE_SCRIPT`, `POSTPROCESS_SCRIPT`.

**Data-generation merge pattern**: If there is a separate one-off data/truth
generation script (e.g. `generate_data.jl`), merge it as a sequential pre-step
inside `setup.sbatch` — run it first, then run the setup/init script. Both are
fast and sequential; bundling them avoids an extra SLURM job and a dependency.
Record `DATA_GEN_SCRIPT` in `hpc_config.sh` (leave empty if absent).

**Adapt script args**: Read the `ARGS` usage in each Julia script carefully —
many pipelines need arguments beyond `output_dir iteration member`. Common extras:
- Forward map: `data_path` (path to the truth/noise file), `eki_path`
- Update: `eki_path` (path to the JLD2 state file), `priors_toml`
Add any extras as variables in `hpc_config.sh` (e.g. `DATA_PATH`, `EKI_PATH`,
`TOML_PATH`) and thread them through the relevant sbatch files.

Read `src/TOMLInterface.jl` to understand the latest directory layout that
`save_parameter_ensemble` and `path_to_ensemble_member` write.

---

### Step 2 — Verify pipeline components and report issues

Check for the following required components and report findings honestly. **Do not
edit the user's Julia files** unless they confirm each fix. Instead, present a
checklist like:

```
✓  Prior built via get_parameter_distribution / constrained_gaussian
✓  Noise Γ estimated and passed to EnsembleKalmanProcess
✓  Algorithm settings: N_ensemble=6, N_iterations=5, process=Inversion()
✓  EKP iteration loop: forward-map per member + update_ensemble!
✓  JLD2 save of eki, prior, param_dict
⚠  run_computer_model.jl copies path_to_ensemble_member locally — import from
    EKP.TOMLInterface instead to avoid drift
⚠  RNG is round-tripped through truth.jld2 per member — this serialises members
    and blocks array parallelism. Fix: generate per-member RNGs from a seed in
    initialize_EKP.jl and pass via member TOML or args.
⚠  save_file inconsistency: "parameters.toml" in init but "parameters" in update
    (both work, but inconsistent and confusing)
```

Flag the following SinusoidInterface-class issues whenever you see them:
1. EKP functions duplicated in the forward-map script (`path_to_ensemble_member`,
   `get_parameter_values`) — risk of drift if the real API changes.
2. Mutable RNG state stored in the shared truth file and re-read/written by every
   member — this creates a race condition under SLURM array parallelism.
3. Inconsistent `save_file` argument between init and update scripts.
4. Legacy `EnsembleKalmanProcess(params, y, Γ, process)` positional constructor
   — still works in v2.7.1 but predates the `Observation` API; worth flagging.
5. Missing `[compat]` in `Project.toml`.

After presenting the checklist, ask: "Should I apply the flagged fixes? I can
apply them all, apply specific ones, or leave the Julia as-is and just generate
the HPC scripts."

---

### Step 3 — Fill and copy the asset templates

Create `<example_dir>/slurm-variant/` and write all 10 files there. Never write
generated HPC files directly into the example root — always use `slurm-variant/`.

Substitute template placeholders (`__SETUP_SCRIPT__`, `__FORWARD_SCRIPT__`,
`__UPDATE_SCRIPT__`, `__POSTPROCESS_SCRIPT__`, `__N_ENSEMBLE__`,
`__N_ITERATIONS__`, `__RUN_DATE__`) with values detected in Step 1 and the
current date.

**Required files — all 10 must be written, no exceptions:**
1. `hpc_config.sh`
2. `precompile.sbatch`
3. `setup.sbatch`
4. `forward_map.sbatch`
5. `update_ensemble.sbatch`
6. `postprocess.sbatch` (write even if POSTPROCESS_SCRIPT is empty — the template exits cleanly)
7. `run_precompile.sh`
8. `run_pipeline.sh`
9. `run_postprocess.sh`
10. `README.md` — **must always be written last, after all other files**. Fill in the `__SETUP_SCRIPT__`, `__FORWARD_SCRIPT__`, `__UPDATE_SCRIPT__`, `__POSTPROCESS_SCRIPT__` placeholders in the README template so the stage table is accurate. Do not skip or defer this file.

**`hpc_config.sh`** is the manifest the user will edit. Pre-fill everything you
know; leave resource dials at sensible defaults. Set `RUN_DATE` to today's date
(`$(date +%Y-%m-%d)` format). Add a comment reminding the user to pin `RUN_DATE`
before submitting an array run so all tasks share the same output directory.
Also add any extra script-arg variables discovered in Step 1 (`DATA_PATH`,
`EKI_PATH`, `TOML_PATH`, etc.).

**`#SBATCH` headers** cannot read shell variables, so the generator must
substitute resource values into the sbatch files at copy time using the detected
`N_ENSEMBLE` value and the defaults from `hpc_config.sh`.

**Iterate: if the user doesn't know their cluster details yet**, leave
`ACCOUNT`, `PARTITION`, `JULIA_MODULE` as clearly-labelled TODO placeholders and
explain where to fill them in (Step 5 summary).

---

### Step 4 — Self-check the generated scripts

After writing all files into `slurm-variant/`, confirm all 10 are present with
`ls slurm-variant/` and flag any missing file before proceeding.

Then:
1. Run `bash -n <file>` on every `.sh` and `.sbatch` in `slurm-variant/` — verify no syntax errors.
2. Run `shellcheck <file>` on each if `shellcheck` is available (`command -v
   shellcheck`). Note: intentional SLURM-specific constructs (e.g. referencing
   `${SLURM_ARRAY_TASK_ID}` defined by the scheduler) can be `# shellcheck
   disable`d.
3. Run `bash slurm-variant/run_pipeline.sh --dry-run` to print the full
   dependency tree without submitting anything. Verify it looks like:
   ```
   [DRY RUN] setup: sbatch --parsable -A <ACCOUNT> setup.sbatch -> JID=...
   [DRY RUN] fwd iter 0: sbatch --parsable --array=1-N --dependency=afterok:...
               --kill-on-invalid-dep=yes --export=ALL,ITERATION=0 forward_map.sbatch
   ...
   [DRY RUN] postprocess: sbatch --parsable --dependency=afterok:... postprocess.sbatch
   ```
4. Confirm `precompile.sbatch` is the **only** file that does NOT set
   `export JULIA_PKG_PRECOMPILE_AUTO=0`. Check every other `.sbatch` file
   for this line — its absence is a correctness bug that will cause 60 concurrent
   array tasks to each try to recompile Julia packages simultaneously.
5. Confirm the output path in every script resolves to
   `$OUTPUT_ROOT/$RUN_DATE/...` and that forward-map member paths match the
   TOMLInterface contract `iteration_<i>/member_<j>/parameters.toml`.

**HPC verification:** Because this agent typically cannot submit to a real cluster,
after completing the self-checks explicitly ask the user:

> "The structural checks pass. Can you verify on your cluster? The sequence is:
> `bash run_precompile.sh` (wait for it to finish), then `bash run_pipeline.sh`.
> After submitting, `squeue -u $USER` should show the full job tree. Happy to help
> interpret any failures."

---

### Step 5 — Summarise and hand off to the user

Produce a concise summary:
- All 10 files written to `slurm-variant/` (list each explicitly so the user can verify)
- Key `slurm-variant/hpc_config.sh` toggles to check before the first run
  (especially `JULIA_MODULE`, `ACCOUNT`, `PARTITION`, `RUN_DATE`, per-stage
  resources, and any extra arg variables like `DATA_PATH`, `EKI_PATH`)
- Launch sequence:
  1. `cd <example_dir>` — scripts use relative paths; run them from here
  2. Edit `slurm-variant/hpc_config.sh` to match your cluster
  3. `bash slurm-variant/run_precompile.sh` (once per environment change)
  4. `bash slurm-variant/run_pipeline.sh` to submit the full dependency tree
  5. `bash slurm-variant/run_postprocess.sh` to re-run postprocessing on existing output
- Any confirmed Julia fixes applied and any outstanding flagged issues not yet
  applied
- Where to look for SLURM logs: `$OUTPUT_ROOT/$RUN_DATE/slurm/`

---

### Step 6 — Offer further improvement

Close by offering to improve the **slurm-pipeline-manager** skill itself via
skill-creator. The user may have ideas from seeing the generated output for the
first time — recurring edge cases, resources that didn't fit their cluster,
aspects of the README that were confusing.

> "Would you like to improve the **slurm-pipeline-manager** skill itself using
> skill-creator? You can share suggestions, or I can analyse patterns from this
> session — recurring edge cases, cluster-specific workarounds, anything that felt
> awkward — to refine the skill for next time."

---

## Reference files

- `references/tomlinterface.md` — condensed TOMLInterface API + TOML prior syntax.
  Read this in Step 1 to understand the `iteration_<i>/member_<j>` directory
  contract and the available helper functions.
- `assets/` — all templates. Each file is a starting point; Step 3 substitutes
  placeholders and writes the result to `<example_dir>/slurm-variant/`. The user
  then owns and edits the files in `slurm-variant/`.
