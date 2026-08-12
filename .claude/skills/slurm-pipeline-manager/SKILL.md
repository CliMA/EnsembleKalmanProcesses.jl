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

This skill takes an EKP Julia example (decoupled or monolithic) and produces a
complete SLURM job-dependency tree inside a `slurm-variant/` subdirectory.
`slurm-variant/` is fully self-contained — it is both the HPC submission
directory and the Julia project root. The original example is never modified.

The generated tree uses a one-shot precompile job followed by a
dependency-chained `setup → forward_map[array] → update_ensemble` loop across
iterations, ending in postprocessing — so ensemble members run in parallel and
every stage pays precompile cost only once.

It also supports **maintenance mode**: if `slurm-variant/hpc_config.sh` already
exists, re-invoking the skill updates files in place without clobbering the
user's manifest settings.

---

## Core principle — slurm-variant/ is fully self-contained

> **Everything new lives in `slurm-variant/`. The original example directory is
> never modified.**

This includes:
- All SLURM / shell scripts (`.sbatch`, `.sh`)
- All decoupled Julia scripts (`initialize_EKP.jl`, `run_forward_model.jl`, etc.)
- Any shared model file (`model.jl` or similar)
- Parameter TOML files (`priors.toml`)
- A `Project.toml` copy with any additional dependencies

`slurm-variant/` is the working directory for every HPC job — users `cd` into it
before running anything. This means `JULIA_PROJECT="."` in `hpc_config.sh` and
script names have no path prefix (e.g. `SETUP_SCRIPT="initialize_EKP.jl"`).

---

## How the generated pipeline is structured

```
precompile ─(manual, run first)

setup ──afterok──► fwd_iter_0[array 1..N] ──afterok──► update_iter_0
      (afterok chain continues for each iteration)
      ──afterok──► fwd_iter_K-1[array 1..N] ──afterok──► update_iter_K-1
                                                         ──afterok──► postprocess
```

Key design choices:
- **One precompile job only.** Every run-stage job sets `JULIA_PKG_PRECOMPILE_AUTO=0`.
  Without this, 60 concurrent array tasks each attempt to precompile — thrashing
  the shared Julia depot and wasting time.
- **Forward map as a SLURM array.** `--array=1-$N_ENSEMBLE` gives one task per
  ensemble member; the iteration index travels via `--export=ALL,ITERATION=$i`.
- **`afterok` + `--kill-on-invalid-dep=yes` everywhere.** If one stage fails the
  whole tree is killed, not left stuck in the queue. EKP errors on < 2 successful
  members (commit 8f2d3fa), so partial-run continuation is not useful.
- **Date-stamped output root.** `RUN_DATE` in `hpc_config.sh` pins all jobs in a
  run to the same output directory.

---

## Workflow

### Mode 0 — Detect generate vs. maintenance

Check whether `<example_dir>/slurm-variant/hpc_config.sh` already exists. If so,
enter **maintenance mode**: read the existing `hpc_config.sh` as the source of
truth, understand the user's requested change, and edit only what is necessary
inside `slurm-variant/` — do **not** overwrite `hpc_config.sh` with fresh
template defaults.

If `slurm-variant/` does not yet exist, proceed with generate mode below.

---

### Step 1 — Read the example and map the pipeline stages

Read every `.jl` file in the named example directory and determine whether the
pipeline is **already decoupled** or **monolithic**.

**Decoupled pipeline** — separate files for each stage:

| Role | What to look for |
|---|---|
| **Setup/init** | builds `prior`, `EnsembleKalmanProcess`, saves `eki`, `param_dict`, `prior` to JLD2, calls `save_parameter_ensemble` for iteration 0 |
| **Forward map** | reads `parameters.toml` from a member dir, runs the model, writes `output.jld2`; accepts `iteration` and `member` as args |
| **Update** | loads `eki.jld2`, loops members collecting `G_ens`, calls `update_ensemble!`, saves next-iteration TOMLs and updated `eki.jld2` |
| **Postprocess** | plotting / analysis — no EKP update; may be absent |
| **Data generation** | creates truth + noise; usually a one-off, may be merged into setup |

Record `SETUP_SCRIPT`, `FORWARD_SCRIPT`, `UPDATE_SCRIPT`, `POSTPROCESS_SCRIPT`.

**Monolithic pipeline** — a single `.jl` file runs the full EKP loop inline
(prior → truth → EKP init → forward loop → update loop → plot). This cannot be
submitted to SLURM as-is. You must split it into the decoupled roles above.
Write ALL split scripts into `slurm-variant/` — never create new `.jl` files in
the original example directory.

When splitting a monolithic script:
- Put shared model code (forward map function, model definition) in a
  `model.jl` (or `<name>_model.jl`) inside `slurm-variant/`; other scripts
  `include()` it. Because `@__DIR__` in Julia resolves to the script's own
  directory, `include("model.jl")` correctly finds the file inside `slurm-variant/`
  regardless of where Julia was launched from.
- Put parameter TOML definitions in `priors.toml` inside `slurm-variant/`.
- Create `slurm-variant/Project.toml` as a copy of the original with any new
  dependencies added (typically `JLD2`, `TOML`). The original `Project.toml`
  is never touched.
- Fix the RNG race: in monolithic scripts the forward map often uses a shared
  module-level RNG. In the SLURM variant each array task runs in its own
  process, so seed each member's RNG deterministically from `(iteration, member)`,
  e.g. `Random.MersenneTwister(iteration * 10_000 + member)`.

**Data-generation merge pattern**: If there is a separate one-off data/truth
generation script (e.g. `generate_data.jl`), merge it as a sequential pre-step
inside `setup.sbatch` — run it first, then the setup/init script. Bundle them to
avoid an extra SLURM job and dependency.

**Adapt script args**: Read the `ARGS` usage in each Julia script carefully —
many pipelines need arguments beyond `output_dir iteration member`. Common extras:
- Forward map: `data_path`, `eki_path`
- Update: `eki_path`, `priors_toml`

Add any extras as variables in `hpc_config.sh` (e.g. `DATA_PATH`, `EKI_PATH`,
`TOML_PATH`) and thread them through the relevant sbatch files.

Read `src/TOMLInterface.jl` to understand the `iteration_<i>/member_<j>/`
directory layout that `save_parameter_ensemble` and `path_to_ensemble_member` use.

---

### Step 2 — Verify pipeline components and report issues

Check for the following required components and report findings honestly. **Do not
edit the user's original Julia files** unless they confirm each fix. Present a
checklist:

```
✓  Prior built via get_parameter_distribution / constrained_gaussian
✓  Noise Γ estimated and passed to EnsembleKalmanProcess
✓  Algorithm settings: N_ensemble=6, N_iterations=5, process=Inversion()
✓  EKP iteration loop: forward-map per member + update_ensemble!
✓  JLD2 save of eki, prior, param_dict
⚠  run_computer_model.jl copies path_to_ensemble_member locally — import from
    EKP.TOMLInterface instead to avoid drift
⚠  RNG is round-tripped through truth.jld2 per member — race condition under
    SLURM array parallelism. Fix: seed per-member RNG from (iteration, member).
⚠  save_file inconsistency: "parameters.toml" in init but "parameters" in update
```

Flag the following SinusoidInterface-class issues whenever you see them:
1. EKP functions duplicated in the forward-map script (`path_to_ensemble_member`,
   `get_parameter_values`) — risk of drift if the real API changes.
2. Mutable RNG state stored in the shared truth file and re-read/written by every
   member — race condition under SLURM array parallelism.
3. Inconsistent `save_file` argument between init and update scripts.
4. Legacy `EnsembleKalmanProcess(params, y, Γ, process)` positional constructor
   — still works in v2.7.1 but predates the `Observation` API.
5. Missing `[compat]` in `Project.toml`.

After presenting the checklist, ask: "Should I apply the flagged fixes? I can
apply them all, apply specific ones, or leave the Julia as-is and just generate
the HPC scripts."

---

### Step 3 — Write all files into slurm-variant/

Create `<example_dir>/slurm-variant/` and write everything there. The complete
file list depends on whether the pipeline was already decoupled or needed splitting:

**Always required (10 SLURM/shell files):**
1. `hpc_config.sh`
2. `precompile.sbatch`
3. `setup.sbatch`
4. `forward_map.sbatch`
5. `update_ensemble.sbatch`
6. `postprocess.sbatch` (write even if `POSTPROCESS_SCRIPT` is empty — template exits cleanly)
7. `run_precompile.sh`
8. `run_pipeline.sh`
9. `run_postprocess.sh`
10. `README.md` — **always written last**, after all other files. Fill in all script-name placeholders so the stage table is accurate.

**Additional files for monolithic-split or dep-extended pipelines:**
- Decoupled Julia scripts (e.g. `initialize_EKP.jl`, `run_forward_model.jl`, `update_EKP.jl`, `postprocess.jl`)
- Shared model file (e.g. `model.jl`)
- `priors.toml` — parameter TOML definitions
- `Project.toml` — copy of the example's `Project.toml` with added deps (`JLD2`, `TOML`, etc.)

**`hpc_config.sh`** is the manifest the user will edit. Key settings:
- `JULIA_PROJECT="."` — `slurm-variant/` is the project root; `.` always resolves correctly
- Script names without any path prefix: `SETUP_SCRIPT="initialize_EKP.jl"` etc.
- Set `RUN_DATE` to today's date. Add a comment reminding the user to pin it before a run.
- Leave `ACCOUNT`, `PARTITION`, `JULIA_MODULE` as clearly-labelled TODO placeholders if unknown.
- Add any extra arg variables discovered in Step 1 (`DATA_PATH`, `EKI_PATH`, etc.).

**`#SBATCH` headers** cannot read shell variables, so substitute resource values
at write time using the detected `N_ENSEMBLE` and the resource defaults.

---

### Step 4 — Self-check the generated scripts

After writing all files, run checks **from inside `slurm-variant/`**:

```bash
cd <example_dir>/slurm-variant/
```

1. `ls` — confirm all required files are present.
2. `bash -n <file>` on every `.sh` and `.sbatch` — verify no syntax errors.
3. `shellcheck <file>` on each if available (`command -v shellcheck`).
4. `bash run_pipeline.sh --dry-run` — verify the dependency tree prints correctly:
   ```
   [DRY RUN] setup: sbatch --parsable -A <ACCOUNT> setup.sbatch -> JID=...
   [DRY RUN] fwd iter 0: sbatch --array=1-N --dependency=afterok:... forward_map.sbatch
   ...
   [DRY RUN] postprocess: sbatch --dependency=afterok:... postprocess.sbatch
   ```
5. Confirm `precompile.sbatch` is the **only** file that does NOT export
   `JULIA_PKG_PRECOMPILE_AUTO=0`. Check every other `.sbatch` for this line — its
   absence causes 60 concurrent array tasks to each try to recompile simultaneously.
6. Confirm `JULIA_PROJECT="."` and no `slurm-variant/` path prefix appears in
   `SETUP_SCRIPT`, `FORWARD_SCRIPT`, `UPDATE_SCRIPT`, or `POSTPROCESS_SCRIPT`.

**HPC verification**: After the local checks, ask the user:

> "The structural checks pass. Can you verify on your cluster? From inside
> `slurm-variant/`: run `bash run_precompile.sh` (wait for it to finish), then
> `bash run_pipeline.sh`. After submitting, `squeue -u $USER` should show the
> full job tree. Happy to help interpret any failures."

---

### Step 5 — Summarise and hand off to the user

Produce a concise summary:
- All files written to `slurm-variant/` (list each explicitly so the user can verify)
- Key `hpc_config.sh` toggles to check before the first run
  (`JULIA_MODULE`, `ACCOUNT`, `PARTITION`, `RUN_DATE`, per-stage resources,
  and any extra arg variables like `DATA_PATH`, `EKI_PATH`)
- Launch sequence:
  1. `cd <example_dir>/slurm-variant/` — this is the HPC home; run everything from here
  2. Edit `hpc_config.sh` to match your cluster
  3. `bash run_precompile.sh` (once per environment change)
  4. `bash run_pipeline.sh` to submit the full dependency tree
  5. `bash run_postprocess.sh` to re-run postprocessing on existing output
- Any confirmed Julia fixes applied and any outstanding flagged issues
- Where to look for SLURM logs: `output/$RUN_DATE/slurm/`

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

- `assets/` — all SLURM/shell templates. Step 3 substitutes placeholders and
  writes the result to `<example_dir>/slurm-variant/`.
