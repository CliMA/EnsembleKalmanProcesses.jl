---
name: error-message-manager
description: >
  Rewrite vague, delayed, or low-context Julia error messages into structured,
  actionable diagnostics. Invoke this skill whenever the user mentions: error
  message, improve errors, rewrite @assert, ArgumentError, DimensionMismatch,
  DomainError, vague error, error rewrite, Julia exception, diagnostic, throw,
  validation, early check, assert to throw, or asks to improve how the code
  fails. Also use it when reviewing code for user-facing clarity, when a user
  says errors are confusing or unhelpful, or when auditing a module for
  low-context exceptions. Use it proactively when you see bare @assert, error("..."),
  or throw(ErrorException(...)) calls in Julia code you are reading or editing.
---

# error-message-manager

Rewrite vague, delayed, or low-context Julia error messages into structured,
actionable diagnostics. The goal is errors that tell the user exactly what went
wrong, what was expected, what was received, and—whenever a likely fix exists—
what to do next. Prefer catching mistakes early (at API boundaries) over letting
them propagate into cryptic numerical failures.

## Workflow

### Step 0 — Offer an Explore agent for multi-file scope

If the user's request covers more than one file — a whole directory, a module, or
the entire repo — offer to spawn an Explore agent before doing any file reads
yourself. The agent runs all the reads in parallel without flooding the main
context, and returns a structured inventory you can act on directly.

**When to offer**: any time the target is a directory path (e.g. `src/`) or a
vague scope like "the whole package" or "all the source files".

**Offer text** (adapt as needed):
> "This spans multiple files — I'd recommend spawning an Explore agent to survey
> all `throw`/`@assert`/`error` sites in parallel. It keeps the audit fast and
> leaves the main context clean for the actual rewrites. Want me to do that?"

**Agent prompt to use** (fill in `<path>` and `<package_name>`):

```
Audit `<path>` for error-raising patterns. For every `@assert`, `error(`, or
`throw(` site in every `.jl` file:

1. Record: file, line number, exception type (or "bare @assert" / "bare error"),
   and the full message text (including multiline strings).
2. Classify message quality:
   - "good"  — has `$(expr)` interpolation showing the actual received value
   - "vague" — missing a received value, or no Expected/Got structure
   - "missing" — bare `@assert` with no message at all
3. Note whether the site is at an API boundary (user-facing input) or an internal
   invariant (would require a package bug to fire).

Return a markdown table with columns:
  File | Line | Exception type | Quality | Notes (one-line note on what's wrong if vague/missing)

Focus only on sites that are "vague" or "missing" — skip "good" ones.
```

**How to use the result**: treat the returned table as your working inventory for
Steps 1 and 2. You do not need to re-read the flagged files yourself to classify —
go straight to reading only the lines that need rewrites (Step 3 onwards).

---

### Step 1 — Audit the target scope

Identify which files or functions to address. If the user named a specific
function, start there. If the request is repo-wide, run:

```
rg -n '(@assert[^(]|@assert\(|error\(string\(|throw\(ErrorException)' src/
```

Then collect all message-less `@assert` calls:

```
rg -n '@assert' src/ | grep -v '"'
```

For each hit, record: file, line, the condition being checked, and whether it
guards user-provided input (API boundary) or an internal invariant.

### Step 2 — Classify each site

Use this table to choose the right exception type:

| Condition | Exception |
|---|---|
| Invalid user-provided argument | `ArgumentError` |
| Array/matrix shape mismatch | `DimensionMismatch` |
| Inconsistent argument types across parameters | `ArgumentError` |
| Mathematically invalid value (negative variance, etc.) | `DomainError` |
| Invalid index | `BoundsError` |
| Internal invariant that should never fire | `error(...)` |
| Missing interface implementation | `MethodError` or structured `ArgumentError` |

Avoid `ErrorException` unless there is no better choice.

**Type mismatches vs dimension mismatches**: an `@assert isa(x_mean, AbstractVector{FT})` inside an
`if isa(x, AbstractMatrix{FT})` branch is checking that the user supplied *consistent* arguments
(matrix ensemble → vector mean), not that two arrays have matching sizes. Use `ArgumentError`, not
`DimensionMismatch`, for this pattern.

Distinguish **API boundary** sites (where the user passed something wrong — prefer
typed exceptions with actionable messages) from **internal invariant** sites
(where a bug in the package itself would have to exist — bare `error(...)` with a
clear note is fine there).

**Double-gated invariants**: if a helper is only ever called after the public API has already
checked the same condition (e.g., `get_vector_of_parameterized` is called from `construct_prior`
only when `d.args[1] == Symbol("VectorOfParameterized")` is true), the check inside the helper is
an internal invariant even though it looks like a user-data check. Use a single-line `error(...)`
rather than a full structured `ArgumentError`:

```julia
# internal invariant — the caller already validated this
d.args[1] == Symbol("VectorOfParameterized") || error(
    "Internal error: get_vector_of_parameterized called with non-VectorOfParameterized expression (got $(d.args[1]))",
)
```

### Step 3 — Rewrite with the canonical layout

Use this structure for every user-facing exception:

```julia
throw(ArgumentError("""
Short one-line summary of the failure.

Expected:
    <what would have been valid>

Got:
    <what was actually received, with interpolated values>

Context:
    <surrounding state that helps locate the problem>

Suggestion:
    <most likely fix>
"""))
```

Section rules:
- **Summary**: always present; one line; imperative or declarative.
- **Expected / Got**: strongly preferred for any mismatch check; use `$(expr)`
  interpolation to show actual values.
- **Context**: include when the same error can arise from multiple call sites and
  naming the calling function or struct helps the user orient.
- **Suggestion**: include whenever a likely fix exists. Omit rather than write a
  generic platitude.
- Never dump full matrices or large arrays. Prefer `size(x)`, `eltype(x)`,
  `typeof(x)`, or a scalar summary statistic.

### Step 4 — Move validation early

If the current code lets an invalid input reach a numerical routine before
failing (e.g., `cholesky` on a non-symmetric matrix, `inv` on a singular one),
add an explicit guard at the API boundary:

```julia
# Before: error surfaces deep in cholesky
cov_chol = cholesky(C)

# After: check at the boundary, raise immediately
issymmetric(C) || throw(ArgumentError("""
Covariance matrix must be symmetric.

Got:
    size(C) = $(size(C))
    norm(C - C') = $(norm(C - C'))

Suggestion:
    Pass a symmetric matrix, e.g. `C = (C + C') / 2`.
"""))
cov_chol = cholesky(C)
```

Use `||` for single-condition guards. For multi-condition guards, use `if/throw`.

When using `||` with a multiline triple-quoted throw, the closing `))` goes on its own line
immediately after the closing `"""`:

```julia
condition || throw(ArgumentError("""
Summary line.

Expected:
    ...

Got:
    ...
"""))   # ← closing )) on the line right after the closing """
```

This is the only layout that keeps indentation correct — triple-quoted strings in Julia do not
strip leading whitespace, so indenting the message body would include those spaces in the string.

### Step 5 — Preserve domain language

Write messages in terms the user understands, not in terms of internal Julia
dispatch or linear algebra internals. For example:

- Say "ensemble member count" not "size(x, 2)"
- Say "parameter covariance matrix" not "the second argument to cholesky"
- Say "observation noise covariance" not "Γ"

### Step 6 — Apply rewrites

Edit each site, keeping the surrounding code untouched. Confirm the package
still loads:

```
julia --project -e 'using EnsembleKalmanProcesses'
```

### Step 7 — Add @test_throws tests

Before writing any test, check whether coverage already exists. Grep the matching
`test/<module>/runtests.jl` for the public API function that reaches the rewritten
site:

```bash
grep -n '@test_throws' test/<module>/runtests.jl | grep '<function_name>'
```

Three outcomes:

| Situation | Action |
|---|---|
| `@test_throws <correct_type>` already present | Skip — do not add a duplicate |
| `@test_throws <wrong_type>` already present | Update the existing line to the new type |
| No coverage at all | Add a new test |

For every site that needs a new test, add it in the matching `test/<module>/runtests.jl`:

```julia
@test_throws ArgumentError multiplicative_inflation!(ekp; s = 2.0)
```

Use the specific exception type — never bare `@test_throws Exception`. The test
should construct the minimal invalid input that triggers the new error, without
duplicating happy-path coverage.

**Update existing tests that used the wrong type.** If the file already has a
`@test_throws ErrorException` (or any other type) for a site you're rewriting to
`ArgumentError`, update that existing test in the same edit. Leaving a stale
`@test_throws ErrorException` will cause it to pass against the old code but fail
once your rewrite lands — or vice versa.

**Testing unexported helpers.** If the site is inside an unexported helper (e.g.,
`construct_constraint`, `construct_2d_array`), do not `import` the internal
directly. Instead, test through the nearest exported public API function that
calls it, using invalid input that propagates to the helper:

```julia
# construct_constraint is unexported — test via get_parameter_distribution
no_constraint_dict = Dict("uq_param" => Dict("prior" => "Parameterized(Normal(0.0, 1.0))"))
@test_throws ArgumentError get_parameter_distribution(no_constraint_dict, "uq_param")
```

This keeps tests coupled to the public contract and avoids brittleness when
internal function names change.

### Step 8 — Offer to improve the skill

Once the rewrites and tests are clean, offer: "Would you like to improve the
**error-message-manager** skill itself using skill-creator? You can share
suggestions, or I can analyse patterns from this session—recurring edge cases,
exception-type decisions, or anything that felt awkward—to refine the skill for
next time."

---

## Style rules

- **Triple-quoted strings** for all multiline messages.
- **No full matrix dumps**. Use `size(x)`, `eltype(x)`, `norm(x - ...)`, or
  `extrema(x)` instead.
- **Interpolate actual values** in Got sections so the user sees the numbers,
  not just variable names. For `String`-typed arguments use `$(repr(x))` rather
  than `$(x)` — it adds the surrounding quotes so the output clearly reads as a
  string value (e.g. `Got: sigma_points = "bad"` instead of `Got: sigma_points = bad`).
- **Raise early**: prefer guarding at the function entry point over deep inside a
  helper.
- **No `@assert` for user-facing validation**. `@assert` is a debugging tool;
  it can be compiled out. Use explicit `throw` instead.
- **Single-line messages are fine** when the failure is unambiguous and no
  Expected/Got context would add clarity.

---

## Canonical before/after examples

### Replace a vague `error(string(...))`

```julia
# Before
if scaled_Δt >= 1.0
    error(string("Scaled time step: ", scaled_Δt, " is >= 1.0", "\nChange s or EK time step."))
end

# After
if scaled_Δt >= 1.0
    throw(ArgumentError("""
Scaled time step exceeds the stability bound.

Expected:
    s * Δt < 1.0

Got:
    s = $s
    Δt = $(get_Δt(ekp)[end])
    s * Δt = $scaled_Δt

Suggestion:
    Reduce the scaling factor `s` or shorten the EK time step.
"""))
end
```

### Replace a bare `@assert` on an API boundary

```julia
# Before
@assert(haskey(param_info, "constraint"))

# After
haskey(param_info, "constraint") || throw(ArgumentError("""
Parameter info dict is missing the required "constraint" key.

Got keys:
    $(collect(keys(param_info)))

Suggestion:
    Ensure the TOML entry for this parameter includes a `constraint = ...` field.
"""))
```

### Replace a single-line string-value error (use `repr`)

```julia
# Before
throw(ArgumentError("sigma_points type is not recognized. Select from \"symmetric\" or \"simplex\". "))

# After
throw(ArgumentError("""
Unrecognized sigma_points type.

Expected:
    "symmetric" or "simplex"

Got:
    sigma_points = $(repr(sigma_points))
"""))
```

Using `repr(sigma_points)` rather than `$(sigma_points)` keeps the string
quotes visible in the output, making it unambiguous that the user passed a
`String` value (and making copy-paste errors easy to spot).

### Replace a dimension-mismatch `@assert`

```julia
# Before
@assert size(x, 2) == length(mean_weights)

# After
size(x, 2) == length(mean_weights) || throw(DimensionMismatch("""
Ensemble size does not match the number of quadrature weights.

Expected:
    size(x, 2) == length(mean_weights)

Got:
    size(x, 2) = $(size(x, 2))
    length(mean_weights) = $(length(mean_weights))
"""))
```

---

## Non-goals

- Do not rewrite every low-level exception in the package. Focus on user-facing
  API boundaries and sites explicitly identified.
- Do not suppress Julia stack traces. The goal is clearer diagnostics, not
  silenced errors.
- Do not add verbosity for its own sake. A short, clear message beats a long,
  generic one.
- Do not expose internal linear algebra variable names or dispatch details when
  domain-level terminology exists.
