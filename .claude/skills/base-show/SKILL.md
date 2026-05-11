---
name: base-show
description: >
  Add concise Base.show and Base.summary methods to Julia types whose default REPL
  representation is unhelpful or overwhelming. Use this skill whenever the user
  mentions that a type prints badly in the REPL, asks to improve how an object is
  displayed or printed, wants a custom show, summary, or repr for a Julia type, or
  says the REPL output is noisy, verbose, or hard to read. Also trigger when the user
  asks to "make the REPL output nicer", "add a show method", "add a summary method",
  "customize display", or "fix what prints when I type a variable name". This skill
  produces compact, informative Base.show and Base.summary methods and matching unit
  tests — invoke it proactively whenever show, summary, display, print, repr,
  or REPL output is mentioned in a Julia context.
---

# base-show

Add concise `Base.show(io::IO, ::MIME"text/plain", x::T)` and `Base.summary(io::IO,
x::T)` methods to Julia types whose default REPL representation is unhelpful or
overwhelming. Julia's default show dumps every field recursively; types that hold
DataFrames, large dictionaries, nested arrays, or many scalar fields produce screens
of unreadable text at the REPL. `Base.summary` provides the compact one-line
description that appears when an object is embedded in a container or shown in a
broader context. This skill produces both methods and accompanying unit tests so that
interactive use of the package is pleasant without losing key summary information.

## Workflow

### Step 1 — Enumerate concrete types

List every concrete (non-abstract) struct defined in the package source:

```
grep -nrE '^(mutable )?struct ' src/
```

Exclude `abstract type` declarations — they cannot be instantiated and do not need
show methods.

### Step 2 — Classify show noisiness

For each concrete type, decide whether its default show output would be noisy. A type
is noisy if it holds at least one of:

- A `DataFrame` or similar tabular collection
- A `Dict` with potentially many entries
- A large or variable-length `Array`
- Another struct that is itself noisy
- More than approximately six fields in total

Also run:

```
grep -nrE 'Base\.(show|summary)' src/
```

Skip any type that already has a custom `Base.show` or `Base.summary` method — do not
overwrite existing customization.

### Step 3 — Write show and summary methods

For each noisy type without existing methods, write **both** a `Base.show` and a
`Base.summary` method.

**`Base.show`** — multi-line detail view used directly at the REPL:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::T)
    println(io, "T")
    println(io, "  field_name : ", summary_value)
    # ...
end
```

The method must:

- Print the type name (and any cheap size hints) on the first line.
- Follow with 1–5 concise summary lines: counts, sizes, or ranges of important fields.
  Never print collection contents.
- Produce at most 10 lines of output for any valid instance, including edge cases such
  as empty collections or zero-element structs.

**`Base.summary`** — single-line description used when the object appears inside a
container or is printed in a broader context (e.g., as an element of a `Vector`):

```julia
function Base.summary(io::IO, x::T)
    print(io, "T (key_hint)")
end
```

The method must:

- Fit on one line — no newlines.
- Convey the most important size or identity hint (e.g., number of elements, key
  dimension), so the reader immediately knows what they are looking at.
- Remain cheap: O(1) field accesses only.

Good examples of what to put in the hint: `"847 basins"`, `"1000×365 grid"`,
`"empty"`. Avoid repeating the type name verbatim as the only content — add value.

**Placement**: place both methods adjacent to their type definition in the same source
file, or gather all show/summary methods in a dedicated `src/show.jl` included from
the main module file. Follow whatever convention is already present in the package;
default to `src/show.jl` if no prior convention exists.

If creating `src/show.jl`, add `include("show.jl")` to the main module file after the
type definitions it references.

### Step 4 — Write unit tests

Write one test block per type, covering both the `show` and `summary` methods. Each
test block must:

- Construct a minimal valid instance of the type.
- For `show`: capture output with `sprint(show, MIME("text/plain"), instance)` and
  assert that it contains the type name and that line count does not exceed 10.
- For `summary`: capture output with `sprint(summary, instance)` and assert that it
  contains the type name and produces exactly one line (no `'\n'` in output).

Avoid asserting exact strings so that cosmetic changes to the output do not break tests.

### Step 5 — Verify

Run the package test suite:

```
julia --project -e 'using Pkg; Pkg.test()'
```

Confirm that all new tests pass and no pre-existing tests regress.

### Step 6 — Offer to improve the skill

After the tests pass and the REPL output looks good, ask the user: "Would you like to improve the **base-show** skill itself using skill-creator? You can suggest changes to the workflow or quality criteria, or I can analyse what came up during this session to identify improvements to the skill."

## Common patterns

### Forward plain show to the MIME method

When a type should look the same whether printed inside a container or directly in the
REPL, add a one-line plain-show forwarder so both display paths converge:

```julia
function Base.show(io::IO, x::MyProcess)
    show(io, MIME("text/plain"), x)
end
```

Without this, `[ekp]` in a Vector will fall back to Julia's default dump while the
direct REPL display uses your custom method.

### Truncate long collections with "… and N more"

When a type holds a variable-length collection, cap the loop to keep output bounded:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistribution)
    n = length(x.name)
    println(io, "ParameterDistribution with ", n, " entr", n == 1 ? "y" : "ies")
    max_show = 8
    for i in 1:min(n, max_show)
        println(io, "  '", x.name[i], "': ", sprint(summary, x.distribution[i]))
    end
    n > max_show && println(io, "  … and ", n - max_show, " more")
end
```

### Conditional fields

Only print a field when it carries information:

```julia
if !isnothing(x.prior_mean)
    println(io, "  prior_dim: ", length(x.prior_mean))
end
```

### Pluralisation in summary

Match English grammar for counts that can be 0 or 1:

```julia
print(io, "Observation (", n, " block", n == 1 ? "" : "s", ", dim=", dim, ")")
```

### Arrow notation for mappings

Use `→` in summary when the type represents a transformation between spaces:

```julia
print(io, "PairedDataContainer (", m_in, "×", n_in, " → ", m_out, "×", n_out, ")")
```

### Unicode in mathematical contexts

Use `×` for matrix dimensions, `→` for transformations, `∞` for unbounded constraints,
and `|u|` for set sizes. These are rendered cleanly in all modern Julia terminals and
communicate mathematical meaning concisely.

```julia
# Constraint summary: Constraint{NoConstraint} (−∞, ∞)
lb = get(bounds, "lower_bound", "-∞")
ub = get(bounds, "upper_bound", "∞")
print(io, "Constraint{$(T)} ($(lb), $(ub))")
```

### Use nameof for parametric type identity

When a type carries a type-parameter that identifies its variant, use `nameof` rather
than printing the full parameterised name:

```julia
# Sampler{Float64} (prior_dim=12) — not the raw Sampler{Float64, ...} dump
print(io, "Sampler{", nameof(get_sampler_type(x)), "} (prior_dim=", length(x.prior_mean), ")")
```

### Section separators in show.jl

When collecting all methods in a dedicated `show.jl`, organise by type family with
aligned comment rulers:

```julia
# ── DataContainers ────────────────────────────────────────────────────────────
# ── Observations ─────────────────────────────────────────────────────────────
# ── EnsembleKalmanProcess ────────────────────────────────────────────────────
```

## Quality criteria

| Criterion | Priority | Definition |
|---|---|---|
| Coverage | High | Every type classified as noisy in Step 2 has both a `Base.show` and a `Base.summary` method. |
| Brevity — show | High | Show output is at most 10 lines for any valid instance, including edge cases. |
| Brevity — summary | High | Summary output is exactly one line (no newlines) for any valid instance. |
| Safety | High | Neither method throws on any valid instance. |
| Allocation-safety | High | All data access is O(1): use `length()`, `size()`, `isempty()`, or `first()` on lazy iterators. Never call `collect()`, `sort()`, `filter()`, or any function that materialises a new collection. |
| Test robustness | Medium | Tests assert structural properties, not exact strings. Cosmetic changes do not break tests. |
| No regression | High | Pre-existing tests continue to pass; no unintended changes to other source files. |

## Formatting rules

- **show signature**: `Base.show(io::IO, ::MIME"text/plain", x::MyType)`
- **show first line**: type name via `println(io, "TypeName")`. Cheap size hints may follow on the same line.
- **show subsequent lines**: indented two spaces for readability.
- **summary signature**: `Base.summary(io::IO, x::MyType)`
- **summary content**: one `print` call (no `println`), type name followed by a parenthesised hint, e.g. `print(io, "MyType (847 basins)")`.
- **No collection contents**: print only counts, sizes, or ranges — never iterate and print elements.
- **No allocations**: use `length()`, `size()`, `isempty()`, and `first()` on lazy iterators such as `values(dict)`. Do not call `collect()`, `sort()`, or any function that copies a collection.
- **Tests — show**: use `sprint(show, MIME("text/plain"), x)` to capture output without side effects.
- **Tests — summary**: use `sprint(summary, x)` to capture the one-line description.

## Examples

### Example 1 — matrix-carrying type (size hint)

```julia
# Scenario: a type wraps a parameter matrix and a forward-model output matrix.

# Before (default Julia show — prints the full matrix)
julia> pdc
PairedDataContainer{Float64}(inputs=DataContainer{Float64}(data=[...50×100 matrix...]),
  outputs=DataContainer{Float64}(data=[...30×100 matrix...]))

# After — custom show method
function Base.show(io::IO, ::MIME"text/plain", x::PairedDataContainer)
    m_in,  n_in  = size(x.inputs.data)
    m_out, n_out = size(x.outputs.data)
    println(io, "PairedDataContainer")
    println(io, "  inputs : ", m_in,  " × ", n_in,  " params × samples")
    println(io, "  outputs: ", m_out, " × ", n_out, " obs × samples")
end

# julia> pdc
# PairedDataContainer
#   inputs : 50 × 100  params × samples
#   outputs: 30 × 100  obs × samples

# After — custom summary (arrow notation for a mapping type)
function Base.summary(io::IO, x::PairedDataContainer)
    m_in, n_in   = size(x.inputs.data)
    m_out, n_out = size(x.outputs.data)
    print(io, "PairedDataContainer (", m_in, "×", n_in, " → ", m_out, "×", n_out, ")")
end

# julia> [pdc]
# 1-element Vector{PairedDataContainer{Float64}}:
#  PairedDataContainer (50×100 → 30×100)
```

### Example 2 — collection-carrying type with truncation

```julia
# Scenario: a type holds N named parameter distributions; N can be large.

# Before (default Julia show — prints every distribution in full)
julia> prior
ParameterDistribution{Parameterized, Constraint{NoConstraint}, String}(
  distribution=[Parameterized(Normal{Float64}(μ=0.0, σ=1.0)), ...],
  constraint=[[Constraint{NoConstraint}(bounds=nothing)], ...],
  name=["amplitude", "length_scale", "noise_var", ...])

# After — custom show with truncation
function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistribution)
    n = length(x.name)
    println(io, "ParameterDistribution with ", n, " entr", n == 1 ? "y" : "ies")
    max_show = 8
    for i in 1:min(n, max_show)
        n_con = length(batch(x)[i])
        println(io, "  '", x.name[i], "': ", sprint(summary, x.distribution[i]),
                " [", n_con, " constraint", n_con == 1 ? "" : "s", "]")
    end
    n > max_show && println(io, "  … and ", n - max_show, " more")
end

# julia> prior
# ParameterDistribution with 3 entries
#   'amplitude'   : Parameterized (Normal) [1 constraint]
#   'length_scale': Parameterized (LogNormal) [1 constraint]
#   'noise_var'   : Parameterized (Uniform) [1 constraint]

# After — summary
function Base.summary(io::IO, x::ParameterDistribution)
    n = length(x.name)
    print(io, "ParameterDistribution (", n, " entr", n == 1 ? "y" : "ies", ")")
end
```

### Example 3 — stateful iterative process

```julia
# Scenario: a mutable struct accumulates state across EKI iterations.

# Before (default Julia show — dumps every matrix stored in the struct)
julia> ekp
EnsembleKalmanProcess{Float64, ...}(u=[50×100 matrix, 50×100 matrix, ...],
  g=[...], Δt=[0.5, 0.5], rng=MersenneTwister(...), N_ens=100, ...)

# After — custom show surfacing iteration progress
function Base.show(io::IO, ::MIME"text/plain", x::EnsembleKalmanProcess)
    n_iter = length(x.u) - 1
    n_par  = size(x.u[1].data, 1)
    println(io, "EnsembleKalmanProcess")
    println(io, "  process    : ", nameof(typeof(x.process)))
    println(io, "  N_ens      : ", x.N_ens)
    println(io, "  N_par      : ", n_par)
    println(io, "  n_iter     : ", n_iter)
    println(io, "  scheduler  : ", nameof(typeof(x.scheduler)))
    println(io, "  accelerator: ", nameof(typeof(x.accelerator)))
end

# julia> ekp
# EnsembleKalmanProcess
#   process    : Inversion
#   N_ens      : 100
#   N_par      : 50
#   n_iter     : 5
#   scheduler  : DefaultScheduler
#   accelerator: DefaultAccelerator

# After — summary capturing what matters most at a glance
function Base.summary(io::IO, x::EnsembleKalmanProcess)
    n_iter = length(x.u) - 1
    print(io, "EnsembleKalmanProcess (", nameof(typeof(x.process)),
          ", N_ens=", x.N_ens, ", ", n_iter, " iter)")
end

# julia> [ekp]
# 1-element Vector{EnsembleKalmanProcess{...}}:
#  EnsembleKalmanProcess (Inversion, N_ens=100, 5 iter)
```

### Unit tests

```julia
@testset "PairedDataContainer show" begin
    pdc = PairedDataContainer(rand(50, 100), rand(30, 100))
    out = sprint(show, MIME("text/plain"), pdc)
    @test occursin("PairedDataContainer", out)
    @test count(==('\n'), out) <= 10
end

@testset "PairedDataContainer summary" begin
    pdc = PairedDataContainer(rand(50, 100), rand(30, 100))
    out = sprint(summary, pdc)
    @test occursin("PairedDataContainer", out)
    @test !occursin('\n', out)    # must be exactly one line
end
```
