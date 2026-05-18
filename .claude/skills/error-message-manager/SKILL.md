---
name: error-message-manager
description: >
  Rewrite vague, delayed, or low-context Julia error messages into structured,
  actionable diagnostics. Invoke this skill whenever the user mentions: error
  message, improve errors, rewrite @assert, ArgumentError, DimensionMismatch,
  DomainError, vague error, error rewrite, Julia exception, diagnostic, throw,
  validation, early check, assert to throw, loop context, catch and rethrow,
  warn string, or asks to improve how the code fails. Also use it when reviewing
  code for user-facing clarity, when a user says errors are confusing or
  unhelpful, or when auditing a module for low-context exceptions. Use it
  proactively when you see bare @assert, error("..."), throw(ErrorException(...)),
  @warn string(...), or catch blocks that do not include the original exception
  in their re-throw in Julia code you are reading or editing.
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
   - "good"  — has `$(expr)` interpolation showing the actual received value, and
     is either short (≤~7 lines) or already in a `_throw_` helper function
   - "long-inline" — message content is good, but the body exceeds ~8 lines and
     the throw is written inline (not in a `_throw_` helper)
   - "vague" — missing a received value, or no Expected/Got structure
   - "missing" — bare `@assert` with no message at all
3. Note whether the site is at an API boundary (user-facing input) or an internal
   invariant (would require a package bug to fire).

Return a markdown table with columns:
  File | Line | Exception type | Quality | Notes (one-line note on what's wrong if vague/missing/long-inline)

Focus only on sites that are "vague", "missing", or "long-inline" — skip "good" ones.
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

Also flag `@warn` calls that use string concatenation instead of interpolation:

```
rg -n '@warn\s+string\(' src/
```

And flag `catch` blocks that discard the original exception when re-throwing:

```
rg -n 'catch\s' src/
```

For each `catch` hit, check whether the subsequent `throw` or `error` call
interpolates the caught variable (e.g. `$e` or `sprint(showerror, e)`). If it
does not, the original exception type and message are silently lost.

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

**Loop-body errors**: if the throw is inside a `for` or `while` loop, treat the
loop index and key per-iteration state as required context. Without this, the user
sees "matrix is not positive definite" with no idea whether it happened on
iteration 2 or iteration 200. Always capture `i` (or the loop variable) and the
state that changed between iterations — the ensemble step count, the parameter
vector being updated, the ensemble member index, etc. For *nested* loops, include
both the outer and inner loop variables — the outer variable says which group or
batch failed; the inner variable says which element within it failed. See the
loop-context example in the Canonical examples section below.

**`catch e` losing the original exception**: when a Julia exception is caught and
a new one is thrown, the new message must include the original exception. If it
does not, the user loses the root cause (e.g. `PosDefException`, `SingularException`)
and has no way to distinguish a code bug from a numerical issue. Use
`sprint(showerror, e)` rather than `$e` alone — it formats the exception type and
message together:

```julia
# anti-pattern — root cause vanishes
catch e
    throw(ArgumentError("Matrix factorization failed."))
end

# correct — root cause preserved
catch e
    throw(ArgumentError("""
Matrix factorization failed.

Caused by: $(sprint(showerror, e))

Suggestion:
    ...
"""))
end
```

Only suppress the original exception if it is a well-known internal Julia error
(e.g. `SingularException`) and you are intentionally providing a higher-level
fallback — and even then, log it at `@debug` level.

**`@warn string(...)` concatenation**: `@warn string("...", x, "...")` is the
warning-side equivalent of `error(string(...))` — it's noisy, hard to read, and
doesn't benefit from Julia's interpolation. Rewrite as `@warn "... $x ..."`.
`@warn` messages that use structured strings are also easier to grep and suppress
selectively.

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

### Step 2.5 — Decide: inline or helper?

Before writing the rewrite, decide whether the error belongs inline or should be
extracted into a `_throw_<what>(...)` helper function.

**When to extract** — pull the error into a helper when either condition holds:

- **Length** (primary trigger): the message body exceeds ~8 lines. Extract
  unconditionally — single call site, non-loop context, no surrounding complexity
  required. A full Expected / Got / Suggestion block almost always crosses this
  threshold. Even a one-off long block left inline establishes a pattern that makes
  entire files hard to scan, and accumulates quickly once a few exceptions are made.
- **Duplication**: the same error shape (same summary line, same Expected / Got /
  Suggestion skeleton) appears at ≥2 call sites. Extract even when each block is
  short — the wording drifts silently over time and the call sites collapse to
  readable one-liners.

Inline is appropriate only for genuinely short messages (≤~7 lines) at a single
call site. The bar for "short" is strict: a message with a Summary, an Expected
section, a Got section, and a Suggestion almost certainly exceeds 7 lines and
belongs in a helper. When in doubt, count — if it doesn't fit in 7 lines, extract.

**Where helpers go**

Default: a `## Error helpers` section at the **bottom of the source file**, above
`end # module`. Keeping helpers near their callers preserves traceability — the
reader sees the throw site, jumps to the bottom of the same file, and finds the
message without switching files.

Promote to a shared `src/ErrorMessages.jl` (or the repo's equivalent top-level
utility file) only when **≥2 different source files** call the same helper. Discover
which file to use by reading the top-level module file (e.g. `src/PackageName.jl`)
for its `include(...)` list — then add `include("ErrorMessages.jl")` as the first
`include` so every subsequent file sees the helpers without any `using`/`import`.

**Naming convention**

```
_throw_<what>(positional_required_facts...; kwargs_for_optional_context...)
```

- Underscore prefix → unexported private helper.
- Verb prefix `_throw_` → the function unconditionally raises; callers know there
  is no return value.
- Suffix describes the failure mode: `_dim_mismatch`, `_missing_keys`,
  `_bad_obs_type`, `_not_iterable`.

**Signature convention**

Pass the facts that are *always* present as positional arguments (the offending
value, the expected vs got summary). Pass *optional* context as keyword arguments
with `nothing` defaults — especially loop context (`index`, `total`, `iter`,
`phase`). Build optional sections inside the helper by checking `isnothing(...)`.
This keeps call sites compact and lets the same helper serve both loop and non-loop
contexts (see the *Helper with optional loop context* canonical example).

**Performance: use `@noinline`**

Prefix every helper with `@noinline`. This prevents Julia from inlining the cold
error path into the surrounding hot code, keeping numerical kernels unaffected:

```julia
@noinline function _throw_x_not_iterable(x; where::Symbol)
    throw(ArgumentError(...))
end
```

**What NOT to do**

- Don't create a catch-all `_throw_arg_error(msg::String)` — that just shifts the
  inline triple-quoted block to another file without any DRY benefit.
- Don't use macros (`@check_dim(...)`) — they're magical and harder to debug than
  plain functions.
- Don't bundle all context into one opaque `context::NamedTuple` — explicit kwargs
  are clearer to call and easier to extend.

### Step 3 — Rewrite with the canonical layout

Use this structure for every user-facing exception:

```julia
throw(ArgumentError("""
Short one-line summary of the failure.

Expected:
    <what would have been valid>

Got:
    <what was actually received, with interpolated values>

Loop context:
    iteration  = $iter (of $n_iter)
    <key per-iteration state variable> = $(summary_of_state)

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
- **Loop context**: include whenever the throw is inside a `for` or `while` loop.
  Always report the loop index and the key state that varies between iterations
  (e.g., the EKI step number, the ensemble member index, or the parameter being
  updated). This is what lets the user reproduce the failure without adding
  `println` debugging. Omit for errors that can only fire at a fixed point in the
  code (before the loop starts or after it ends).
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
- **Loop state in Got / Loop context**: when a throw is inside a `for` or `while`
  loop, always name the iteration index and the key state from that iteration.
  A "convergence failed" message without the step count forces the user to add
  `println` debugging to reproduce the failure. When the same loop-context check
  recurs at multiple sites, the loop variables become optional kwargs on a
  `_throw_<what>` helper — see the *Helper with optional loop context* canonical
  example.
- **Preserve the original exception in `catch` blocks**: if you catch `e` and
  throw a new exception, include `$(sprint(showerror, e))` in the new message.
  Dropping `e` silently discards the root cause.
- **`@warn` with interpolation, not `string()`**: replace `@warn string("x=", x)`
  with `@warn "x = $x"`. String concatenation in warnings is harder to read and
  grep.
- **Single-line messages are fine** when the failure is unambiguous and no
  Expected/Got context would add clarity.
- **Extract into `_throw_<what>(...)` helpers** whenever the message body exceeds
  ~8 lines, or when the same Expected / Got / Suggestion skeleton appears at ≥2
  call sites (even if short). A full Expected / Got / Suggestion block nearly always
  exceeds 8 lines and must be a helper — inline is only appropriate for ≤~7-line
  messages at a single call site. Place the helper in a `## Error helpers` section
  at the bottom of the source file; promote to a shared `src/ErrorMessages.jl` only
  when ≥2 different source files share the helper. Use `@noinline`, positional args
  for required facts, and `nothing`-defaulted kwargs for optional context such as
  loop indices. Render each optional section only when its kwarg is non-`nothing`.

---

## Canonical before/after examples

> **Length rule applies to all examples below.** Each example shows the canonical
> message *format* (Expected / Got / Suggestion sections, interpolation, etc.). When
> the message body exceeds ~8 lines — which a full Expected + Got + Suggestion block
> almost always does — the throw must go in a `_throw_<what>(...)` helper per
> Step 2.5, not inline. The first example below models this explicitly. Subsequent
> examples show the message body format; apply the same helper extraction whenever
> the resulting message exceeds 7 lines.

### Replace a vague `error(string(...))`

The after-message has 10 lines (Summary + Expected + Got×3 + Suggestion×2), so it
goes into a `_throw_` helper — extract unconditionally at this length even though
there is only one call site.

```julia
# Before
if scaled_Δt >= 1.0
    error(string("Scaled time step: ", scaled_Δt, " is >= 1.0", "\nChange s or EK time step."))
end

# After — helper in the ## Error helpers section at the bottom of the file
@noinline function _throw_scaled_step_too_large(s, Δt, scaled_Δt)
    throw(ArgumentError("""
Scaled time step exceeds the stability bound.

Expected:
    s * Δt < 1.0

Got:
    s = $s
    Δt = $Δt
    s * Δt = $scaled_Δt

Suggestion:
    Reduce the scaling factor `s` or shorten the EK time step.
"""))
end

# Call site collapses to a single guard line:
scaled_Δt < 1.0 || _throw_scaled_step_too_large(s, get_Δt(ekp)[end], scaled_Δt)
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

### Preserve the original exception when catching and re-throwing

```julia
# Before — PosDefException or SingularException silently discarded
try
    cov_chol = cholesky(cov_u)
catch e
    error("Covariance matrix factorization failed.")
end

# After — root cause preserved, matrix state shown
try
    cov_chol = cholesky(cov_u)
catch e
    throw(ArgumentError("""
Covariance matrix factorization failed during empirical Gaussian sampling.

Got:
    size(cov_u)    = $(size(cov_u))
    isposdef(cov_u) = $(isposdef(cov_u))

Caused by: $(sprint(showerror, e))

Suggestion:
    The ensemble may have collapsed. Pass a non-zero `inflation` keyword
    argument to regularise the sample covariance.
"""))
end
```

`sprint(showerror, e)` formats as `"LinearAlgebra.PosDefException: matrix is not
Hermitian; Cholesky factorization failed."` — far more informative than `string(e)`.
Only suppress `e` when you are intentionally providing a higher-level fallback (e.g.
falling back to `pinv`) and still emit it at `@debug` level.

### Rewrite `@warn string(...)` to use interpolation

```julia
# Before
@warn string("Sample covariance matrix over ensemble is singular.", "\n Applying variance inflation.")

# After
@warn "Sample covariance matrix over ensemble is singular — applying variance inflation."

# Before (with values)
@warn string("More than 50% of runs produced NaNs ($(length(failed_ens))/$(size(g, 2))).", "\nIterating...")

# After
@warn "More than 50% of forward model evaluations produced NaN ($(length(failed_ens))/$(size(g, 2))). Iterating, but consider improving model stability."
```

### Add loop context to an error thrown inside an iteration loop

```julia
# Before — user sees "Cholesky factorization failed" with no idea when
for i in 1:N_iter
    try
        cov_chol = cholesky(C_i)
    catch e
        error("Cholesky factorization failed")
    end
end

# After — guard before cholesky, expose iteration index and diagnostic state
for i in 1:N_iter
    isposdef(C_i) || throw(ArgumentError("""
Covariance matrix is not positive definite at EKI iteration $i.

Expected:
    A positive-definite covariance matrix at every iteration.

Got:
    iteration       = $i / $N_iter
    size(C_i)       = $(size(C_i))
    minimum eigval  = $(minimum(eigvals(Symmetric(C_i))))

Suggestion:
    Ensemble collapse can cause this near iteration $i. Consider adding
    covariance inflation (`multiplicative_inflation!`) or reducing the step size.
"""))
    cov_chol = cholesky(C_i)
end
```

Key points:
- **Move the guard before the failing call** so the message fires with the full
  iteration state still in scope. Catching a `PosDefException` after the fact and
  re-throwing loses the iteration index and the matrix state.
- **Report the loop variable** (`i`, `n`, `iter`) and its upper bound so the user
  knows whether the failure is early (step 2/200, likely a bad initial state) or
  late (step 198/200, likely ensemble collapse).
- **Include one diagnostic scalar** — the minimum eigenvalue, the norm of the
  update step, the ensemble spread — rather than dumping the full matrix.

When this same loop-context error needs to be thrown at multiple sites, the loop
variables (`i`, `N_iter`) naturally become optional kwargs on a `_throw_<what>`
helper. The call site stays a single line and the loop-awareness travels with the
helper everywhere it is used — see the *Helper with optional loop context* example
below.

### Extract a duplicated error into a helper

`transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained`
in `src/ParameterDistributions.jl` each validate the same two preconditions on their
iterable argument `x`. Before extraction, byte-for-byte identical 12-line blocks
appear at both call sites:

```julia
# Before — same two blocks in both transform functions (×2 each = 4 copies total)

# in transform_constrained_to_unconstrained:
if !hasmethod(iterate, [typeof(x)])
    throw(ArgumentError("""
transform_constrained_to_unconstrained: `x` is not iterable.

Expected:
    AbstractVecOrMat or an iterable of AbstractVecOrMat elements (one per EK iteration)

Got:
    $(typeof(x))

Suggestion:
    Pass a Vector or Matrix, or a collection of Vectors/Matrices.
"""))
end
if !isa(x[1], AbstractVecOrMat)
    throw(ArgumentError("""
transform_constrained_to_unconstrained: elements of `x` are not AbstractVecOrMat.

Expected:
    An iterable whose elements are AbstractVecOrMat

Got:
    element type = $(typeof(x[1]))

Suggestion:
    Pass a collection of Vectors or Matrices (one per EK iteration).
"""))
end

# in transform_unconstrained_to_constrained: byte-for-byte identical except the
# summary line reads "transform_unconstrained_to_constrained" instead.
```

After extraction, both functions call two helpers defined once in a `## Error helpers`
section at the bottom of the file:

```julia
# After — helpers at the bottom of src/ParameterDistributions.jl

## Error helpers

@noinline function _throw_x_not_iterable(x; where::Symbol)
    throw(ArgumentError("""
$where: `x` is not iterable.

Expected:
    AbstractVecOrMat or an iterable of AbstractVecOrMat elements (one per EK iteration)

Got:
    $(typeof(x))

Suggestion:
    Pass a Vector or Matrix, or a collection of Vectors/Matrices.
"""))
end

@noinline function _throw_x_elements_not_vecormat(x; where::Symbol)
    throw(ArgumentError("""
$where: elements of `x` are not AbstractVecOrMat.

Expected:
    An iterable whose elements are AbstractVecOrMat

Got:
    element type = $(typeof(x[1]))

Suggestion:
    Pass a collection of Vectors or Matrices (one per EK iteration).
"""))
end

# Both call sites now collapse to two readable guard lines each:
function transform_constrained_to_unconstrained(pd::ParameterDistribution, x)
    hasmethod(iterate, [typeof(x)]) ||
        _throw_x_not_iterable(x; where = :transform_constrained_to_unconstrained)
    isa(x[1], AbstractVecOrMat) ||
        _throw_x_elements_not_vecormat(x; where = :transform_constrained_to_unconstrained)
    # ... algorithm body visible immediately ...
end

function transform_unconstrained_to_constrained(pd::ParameterDistribution, x)
    hasmethod(iterate, [typeof(x)]) ||
        _throw_x_not_iterable(x; where = :transform_unconstrained_to_constrained)
    isa(x[1], AbstractVecOrMat) ||
        _throw_x_elements_not_vecormat(x; where = :transform_unconstrained_to_constrained)
    # ... algorithm body visible immediately ...
end
```

Key points:
- The `where::Symbol` kwarg embeds the calling function name in the message so
  diagnostics stay specific even though the body is shared. Pass a `Symbol` literal
  (`where = :my_func`) — symbols are cheap and render cleanly with `$where`.
- Both functions now have two one-line guards instead of two 12-line blocks; the
  algorithm body is immediately visible.
- `@noinline` keeps the error path out of the hot function body.
- The helpers live at the bottom of the same file — one jump away, no new file.

### Helper with optional loop context

The `for pdd in param_dist_dict_array` loop in `src/ParameterDistributions.jl`
validates each parameter dict but currently reports no position — the user sees
"missing required keys" with no idea which dict in the array triggered the error.
Extracting into a helper adds the index and makes the same helper reusable wherever
that validation appears:

```julia
# Before — inline block, no loop index in the message
for pdd in param_dist_dict_array
    if !all(["distribution", "name", "constraint"] .∈ [collect(keys(pdd))])
        throw(ArgumentError("""
Parameter dictionary is missing required keys.

Expected keys:
    "distribution", "name", "constraint"

Got keys:
    $(sort(collect(string.(keys(pdd)))))

Suggestion:
    Ensure each parameter dict contains all three required keys.
"""))
    end
end

# After — helper with optional loop context at the bottom of the file

@noinline function _throw_param_dict_missing_keys(got_keys; index = nothing, total = nothing)
    loop_ctx = isnothing(index) ? "" : """

Loop context:
    dict index = $index (of $total)"""
    throw(ArgumentError("""
Parameter dictionary is missing required keys.$loop_ctx

Expected keys:
    "distribution", "name", "constraint"

Got keys:
    $got_keys

Suggestion:
    Ensure each parameter dict contains all three required keys.
"""))
end

# Call site — loop now reports position:
for (i, pdd) in enumerate(param_dist_dict_array)
    all(["distribution", "name", "constraint"] .∈ [collect(keys(pdd))]) ||
        _throw_param_dict_missing_keys(
            sort(collect(string.(keys(pdd)))); index = i, total = length(param_dist_dict_array),
        )
end

# The same helper works outside a loop — omit the kwargs and the Loop context
# section is silently suppressed:
all(["distribution", "name", "constraint"] .∈ [collect(keys(pdd))]) ||
    _throw_param_dict_missing_keys(sort(collect(string.(keys(pdd)))))
```

Key points:
- `index` and `total` default to `nothing`; the `Loop context:` section is
  rendered only when they are provided. No special-casing at any call site.
- Switching `for pdd in ...` to `for (i, pdd) in enumerate(...)` is the only
  loop-side change needed to expose the index.
- The user now knows *which* dict failed, not just that one of them did.
- The same helper can be called from a non-loop site (e.g. single-dict validation)
  with zero kwargs and produces a clean message without a Loop context section.

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
- Do not extract truly short errors (≤~7 lines) at a single call site — the
  inline form is easier to grep and keeps cause and message co-located. A summary
  plus a single Got line is a natural ceiling for inline: anything that also includes
  Expected and Suggestion sections almost always exceeds 7 lines and must be a helper.
