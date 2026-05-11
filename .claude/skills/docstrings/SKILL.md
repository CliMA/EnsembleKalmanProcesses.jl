---
name: docstrings
description: >
  Add or normalise Julia docstrings on public symbols (exported types, functions,
  and constants) so the package's public API is fully self-documenting and the
  Documenter.jl docs build passes its checkdocs check.
  Invoke this skill whenever the user mentions: docstring, missing doc,
  undocumented symbol, API doc, checkdocs warning, or asks to document a type or
  function. Also use it when the user asks to "write docs for" or "add docs to"
  source files, or when a CI failure mentions missing or incomplete docstrings.
---

# docstrings

Add or normalise Julia docstrings on public symbols (exported types, functions,
and constants) across the package source. The goal is complete, consistent API
documentation that renders correctly under Documenter.jl and follows whichever
docstring convention is already established in the package — typically
DocStringExtensions macros such as `$(TYPEDEF)`, `$(TYPEDFIELDS)`, and
`$(TYPEDSIGNATURES)`. Completing this skill makes the package's public API fully
self-documenting and satisfies any `checkdocs` requirement in the docs build.

## Workflow

### Step 1 — Detect the existing convention

Read 1–2 symbols that already have complete docstrings to calibrate style.
Identify:

- Whether DocStringExtensions macros are used, and which ones.
- How prose is structured relative to macro-generated content (e.g. does prose
  come before or after `$(TYPEDFIELDS)`?).
- What field documentation pattern is preferred: inline string literals above
  each struct field vs. a separate prose block.

This detected baseline becomes the style target for every new or normalised
docstring. Do not impose a different convention — match what is already there.

### Step 2 — Enumerate candidates

Discover the package name from `Project.toml` (the `name =` field). Then run:

```
grep -nE '^(function |struct |abstract type |mutable struct |const )' src/**/*.jl
```

Cross-reference the result with the module's exported names by reading the main
module file for `export` statements. For each exported symbol, check whether a
non-empty docstring immediately precedes the definition.

Produce a prioritised list:
1. Missing entirely.
2. Empty or stub (e.g. only a macro line with no prose).
3. Incomplete (prose present but key sections absent).

### Step 3 — Draft docstrings

For each candidate, write a docstring that matches the detected convention:

- Use the same macro set as the best-documented symbols already in the package.
- Preserve any inline field string literals already present above struct fields —
  do not merge them into the struct-level docstring.
- Prose should answer: what does this symbol represent or do, when would a caller
  use it, and what are the physical units of key quantities.
- Do not duplicate content that macros generate automatically (e.g. do not
  restate field types when `$(TYPEDFIELDS)` already renders them).
- Physical quantities: always include units in square brackets, e.g. `[m/day]`.
- For functions with more than two arguments, or whose argument semantics are
  not obvious from the name alone, add a `# Arguments` section listing each
  parameter as `` - `name`: description [unit if applicable] ``.
- For every non-trivial public function where a minimal runnable example can be
  written, add a `# Examples` section with a `jldoctest` block so Documenter.jl
  can verify the example stays correct as the code evolves.

### Step 4 — Verify

Find the package name from `Project.toml`, then confirm the package loads
without error:

```
julia --project -e 'import Pkg; Pkg.instantiate(); using <PackageName>'
```

If a docs build is configured (`docs/make.jl` is present), run it and resolve
any `checkdocs` warnings introduced by the new docstrings.

### Step 5 — Offer to improve the skill

Once the docs build is clean, ask the user: "Would you like to improve the **docstrings** skill itself using skill-creator? You can share suggestions, or I can analyse patterns from this session — recurring edge cases, formatting decisions, or anything that felt awkward — to refine the skill for next time."

## Formatting rules

These rules encode the conventions most Julia packages following DocStringExtensions
expect. Apply them consistently.

- **Triple-quoted strings** for all docstrings.
- **First line**: concise one-line summary — imperative mood for functions
  (`"Return the..."`, `"Compute..."`), noun phrase for types and constants.
- **Second line**: blank.
- **Body**: prose, then any macro invocations. `$(TYPEDSIGNATURES)` must be the
  very first line of a function docstring and is the sole source of the method
  signature — never write a manual indented signature as well.
- **No trailing whitespace** inside the docstring.
- **No emojis.**
- **Physical units** in square brackets: `[m/day]`, `[kg/m³]`, `[day]`, etc.
- **Field string literals** (the string above each struct field) are distinct
  from the struct-level docstring. Preserve both; do not merge them.
- Field string literals must describe the field's *semantic role*, not its type.
  Never write a type name inside brackets (e.g. `"[Date]"`, `"[Dict]"`) —
  `$(TYPEDFIELDS)` already renders the type. Reserve square-bracket notation
  exclusively for physical units.
- Avoid vague labels such as "data object" or "container". Say what the field
  represents in domain terms (e.g. "mapping of basin ID to forcing timeseries"
  rather than "dictionary of forcing timeseries data objects").
- **`# Arguments` section**: add after the opening prose for any function with
  more than two parameters, or where argument semantics are non-obvious. Format:
  `` - `name`: description [unit] ``.
- **`# Examples` section**: add for every non-trivial public function where a
  minimal runnable example is feasible. Use `jldoctest` blocks with `julia> `
  prompts and include expected output.
- In every `jldoctest` block, separate each `julia> ` prompt from the next with
  a blank line. Documenter.jl rejects blocks where two prompts appear
  consecutively without an intervening blank line. If a statement produces no
  output, end it with a semicolon and add a blank line before the next prompt.
- If the doctest references any name from the package, the first statement must
  be `julia> using <PackageName>` (followed by a blank line). Do not assume the
  package is already in scope.

## Quality criteria

| Criterion | Weight | What to check |
|---|---|---|
| **Completeness** | High | Every exported symbol has a non-empty docstring after the task is applied. |
| **Convention parity** | High | New docstrings use the same macro set and structural pattern as the best-documented symbols already present. |
| **Informativeness** | Medium | Prose answers "what, when, why". Units present for physical quantities. `# Arguments` section present where needed. `# Examples` jldoctest block present for non-trivial public functions. |
| **No duplication** | Medium | Prose does not duplicate macro-generated content. Field string literals do not restate the field's type. |
| **Correctness** | High | Package loads without error; docs build (if configured) completes without new warnings. |

## Example

The example below is intentionally synthetic — generic type and function names
are used so the pattern stays clear regardless of what the package defines.

```julia
## Before — struct with no docstring; field strings use type names in brackets (anti-pattern)

struct MyStruct
    "Wave velocity [Float64]"   # BAD: [Float64] is the type, not a unit
    velocity::Float64
    "Start date [Date]"         # BAD: [Date] is the type, not a unit
    start::Date
    "Number of nodes"
    n_nodes::Int
end

## After — struct-level docstring added; field strings use units, not type names

"""
    MyStruct

Represent the spatial discretization parameters for a single channel reach.

$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct MyStruct
    "Wave velocity [m/day]"              # GOOD: physical unit
    velocity::Float64
    "Start of the simulation period"     # GOOD: semantic description, no redundant type
    start::Date
    "Number of nodes along the reach"
    n_nodes::Int
end


## Before — function with an empty stub docstring

"""
$(TYPEDSIGNATURES)
"""
function advance(x::MyStruct, dt::Float64)
    ...
end

## After — prose, Arguments, and Examples sections added

"""
$(TYPEDSIGNATURES)

Advance `x` by one time step of length `dt` [days] and return the updated state.

# Arguments
- `x`: current state to advance.
- `dt`: time step length [days].

# Examples
```jldoctest
julia> using MyPackage

julia> m = MyStruct(1.0, Date(2000, 1, 1), 10);

julia> advance(m, 0.5)
...
```
"""
function advance(x::MyStruct, dt::Float64)
    ...
end
```
