---
name: math-auditor
description: >
  Run an adversarial mathematical-accuracy review of a Julia package's src/ and
  test/ directories, producing a dated markdown report plus concise, self-contained
  fix-prompt markdowns suitable for handing to a smaller model (e.g. Sonnet) in a
  later session. Use this skill whenever the user asks for an adversarial review,
  a math audit, a full code review focused on mathematical/numerical correctness,
  a check of algorithmic consistency with the literature, or says things like
  "review the math in src/", "is the algebra right?", "audit the update equations",
  "check the statistics/linear algebra for bugs", or "construct a code review as
  markdown". Trigger even when the user does not say "audit" — any request for a
  correctness-focused sweep of a scientific Julia codebase qualifies.
---

# Math Audit

Adversarial review of a scientific Julia package for **mathematical accuracy and
consistency** — not software architecture (flag architecture only when it causes
mathematical wrongness, e.g. mutation aliasing, accidental type demotion, or
inconsistent conventions between modules).

The output is written for the package's own developers: findings must cite exact
`file:line`, state the correct mathematics, and give a concrete failure scenario.
A finding that can't survive an attempt at refutation doesn't ship.

## What "adversarial" means here

Each reviewer's job is to *break* the code, not describe it. Concretely, hunt for:

- **Wrong equations**: update formulas, gradients, covariances, likelihoods that
  differ from the cited papers or from the docstring's own LaTeX. Derive the
  correct expression independently and diff it against the code.
- **Convention drift**: rows-vs-columns for ensemble members, `N-1` vs `N`
  normalization, factor-of-2 / sign errors, Cholesky `L` vs `U`, covariance vs
  precision, whether noise is added in obs-space or transformed space —
  especially *inconsistencies between modules that must agree*.
- **Statistical validity**: is added noise sampled with the right covariance and
  scaling (e.g. `Δt` scaling in stochastic dynamics)? Are means/covariances
  computed over the right dimension? Deterministic vs stochastic variants
  actually equivalent in expectation?
- **Numerical soundness**: unguarded `inv`/`\` on possibly-singular matrices,
  loss of symmetry/PSD-ness, subtraction-based variance formulas, missing
  regularization, `sqrt` of negative-by-roundoff eigenvalues.
- **Edge cases the math must survive**: ensemble size 1–2, dimension 1 (scalar
  vs matrix degeneracy), zero variance, NaN/failed ensemble members, empty
  minibatches.
- **Test-math consistency**: do the tests actually pin the mathematics
  (analytic solutions, invariants, convergence rates), or just check shapes and
  "it runs"? A wrong equation whose test only checks `size()` is a *double*
  finding: the bug and the missing test.

## Workflow

### 1. Partition

List `src/*.jl` and `test/**/*.jl` with line counts. Group into 4–8 review
units of roughly comparable size, pairing each source module with the tests
that exercise it. Group modules that *share mathematical conventions* together
(e.g. all Kalman-update variants in one or two units) so the reviewer can catch
cross-module inconsistencies.

### 2. Fan out reviewers (parallel agents)

Spawn one agent per unit, in a single message so they run concurrently. Each
agent prompt must include:

- the exact file list for its unit,
- the "What adversarial means here" hunting list above (copy it in — agents
  don't see this skill),
- instructions to check code against docstrings/comments *and* against the
  standard form of the algorithm from the literature,
- a required output format: a JSON-like list of findings, each with
  `file`, `line`, `severity` (critical / major / minor / hygiene),
  `claim` (one sentence), `evidence` (the code vs the correct math),
  `failure_scenario` (concrete inputs → wrong output),
  `verified` (`numerical` / `inspection`), and
  `suggested_fix` (optional, a few lines).

Tell agents explicitly:

- "Prefer few, well-evidenced findings over many speculative ones — but do
  report genuine minor inconsistencies. If the module's math is correct, say so
  and note the strongest invariants the tests pin."
- "When a finding concerns a fixed point, a statistical scaling, or a crash,
  verify it numerically in a scratch script if cheap — a small linear-Gaussian
  fixed-point run, a quick Monte Carlo of the statistic, or reproducing the
  error — and tag it `verified: numerical`. Numerically verified findings are
  worth far more than inspection-only ones." (In one audit, the agents that ran
  code delivered the critical finding pre-verified to 1e-16; the only refuted
  claim of the run came from an inspection-only unit.)
- "Before claiming anything is 'silent' or 'has no warning/guard', grep for
  `@warn`, `@error`, and `throw` at the *constructors and call sites* of the
  code path, not just the function you are reading — guards often live at
  construction time."

As each agent's report arrives, save its raw findings verbatim to a scratchpad
file (one per unit). A full audit plus verification is long enough that
context summarization mid-run can silently lose findings; the scratchpad files
are the durable record the report is assembled from.

### 3. Verify

For each critical/major finding, attempt refutation before it enters the
report: re-read the cited lines yourself, re-derive the math, and check whether
a test or an upstream transformation already accounts for it (common false
positives: a transpose hidden in a helper, normalization done at construction
time, a convention documented elsewhere, a `@warn` at the constructor that the
reviewer never read — re-run the warn/throw grep yourself for any "silent"
claim). Prioritise findings tagged `verified: inspection`; numerically verified
ones usually need only a sanity re-read. Spawn skeptic agents for findings you
can't settle from the main context. Demote or drop findings that don't survive;
mark surviving ones **CONFIRMED** vs **PLAUSIBLE** (couldn't fully verify).

Then build a small **conventions matrix** from the unit reports before writing
anything: rows = modules, columns = the conventions the units commented on
(covariance normalization N vs N−1, Δt/noise-scale placement, order of
regularization vs localization, RNG threading, rows-vs-columns). Any mismatched
cell between modules that must agree is a finding candidate in itself — in
practice the worst bugs are a scale factor applied to one block or module but
not its sibling, and they only become visible side by side.

### 4. Write the report

Create `full-code-review/<YYYY-MM-DD>/` (date from `date +%F`, never from
memory). Write `review.md`:

```markdown
# Adversarial Mathematical Review — <Package> (<date>)
## Scope and method          <!-- files covered, units, verification policy -->
## Summary table             <!-- ID | severity | verdict | file:line | one-line claim -->
## Critical findings         <!-- full detail: evidence, math, failure scenario, fix sketch -->
## Major findings
## Minor findings & hygiene  <!-- terser -->
## Cross-module consistency notes
## Test-coverage gaps        <!-- where math is unpinned by tests -->
## What was checked and found sound   <!-- credit where due; prevents re-auditing -->
```

Findings get stable IDs used everywhere, including fix prompts: `C1…` critical,
`M1…` major, `m1…` minor, `h1…` hygiene; grouped-minor fix prompts get `G1…`.

### 5. Write fix prompts

For each finding with an actionable fix (usually critical + major, plus grouped
minors), write `full-code-review/<date>/fix-prompts/<ID>-<slug>.md`. These are
consumed by a *smaller model in a fresh session with no context*, so each must
be self-contained:

```markdown
# Fix <ID>: <one-line title>
**File**: `src/Foo.jl`, function `bar!`, around line NNN.
**Problem**: <2–4 sentences: what the code does vs what the math requires.
Include the incorrect snippet verbatim.>
**Required change**: <exact edit, or precise description with the correct formula>
**Do not**: <guardrails — e.g. "do not change the API", "do not touch other methods">
**Verify**: <the test to run or add, with the invariant it should pin>
```

Keep each under ~40 lines. One finding per file; group only truly mechanical
repeats (e.g. the same typo pattern in five docstrings) into one prompt.
Quote enough of the offending snippet that the fixer can locate it by function
name + snippet — line numbers drift between the audit and the fix session, so
present them as hints, not anchors.
Also write `fix-prompts/README.md` listing prompts in recommended application
order (independent fixes first, same-file prompts sequenced, conflicting ones
flagged) and naming which fixes *intentionally change numerical results* — so
the fixer checks a failing loose regression test against the analytic reference
in the prompt before "fixing" the test.

### 6. Report back

Final message: lead with the headline (how many confirmed critical/major
findings and the single worst one), then the report path, then a compact
summary table. Do not paste the whole report into the chat.

## Calibration

- Severity: **critical** = produces mathematically wrong results in mainstream
  use; **major** = wrong in common configurations or silently degrades
  statistical properties; **minor** = wrong in edge cases, misleading docs
  math, dead/misnamed math; **hygiene** = style-level (only if math-adjacent).
- A docstring–code mismatch is a real finding even when the code is right —
  users implement against docstrings.
- A statistically unjustified combination that the package explicitly warns
  about (e.g. a constructor `@warn "... experimental ..."`) caps at **minor**
  unless the warning itself is wrong — the trap isn't silent.
- Don't pad. If a unit is sound, the report says so in one paragraph; an audit
  that cries wolf gets ignored next time.

## Improving this skill

After delivering the report, offer: "Would you like to improve the
**math-auditor** skill itself using skill-creator? You can share suggestions, or
I can analyse this run — finding quality, false-positive rate, fix-prompt
usability — to refine the skill for next time."
