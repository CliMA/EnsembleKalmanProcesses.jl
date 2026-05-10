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

## Example

```julia
# Scenario: a type holds a large Dict of DataFrames and a date range.

# Before (default Julia show — dumps the entire Dict contents)
julia> env
MyEnvironment(data=Dict{String, DataFrame}(...dozens of rows...))

# After — custom show method (multi-line REPL display)
function Base.show(io::IO, ::MIME"text/plain", x::MyEnvironment)
    println(io, "MyEnvironment")
    println(io, "  n_basins  : ", length(x.data))
    println(io, "  date_range: ", x.start_date, " to ", x.end_date)
end

# julia> env
# MyEnvironment
#   n_basins  : 847
#   date_range: 2000-01-01 to 2020-12-31

# After — custom summary method (one-line description used inside containers)
function Base.summary(io::IO, x::MyEnvironment)
    print(io, "MyEnvironment (", length(x.data), " basins)")
end

# julia> [env]
# 1-element Vector{MyEnvironment}:
#  MyEnvironment (847 basins)

# Corresponding unit tests
@testset "MyEnvironment show" begin
    env = MyEnvironment(...)          # construct a minimal valid instance
    out = sprint(show, MIME("text/plain"), env)
    @test occursin("MyEnvironment", out)
    @test count(==('\n'), out) <= 10
end

@testset "MyEnvironment summary" begin
    env = MyEnvironment(...)
    out = sprint(summary, env)
    @test occursin("MyEnvironment", out)
    @test !occursin('\n', out)        # must be exactly one line
end
```
