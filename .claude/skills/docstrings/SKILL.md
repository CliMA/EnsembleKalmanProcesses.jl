---
name: docstrings
description: >
  Add or normalise Julia docstrings on public symbols (exported types, functions,
  and constants) so the package's public API is fully self-documenting and the
  Documenter.jl docs build passes its checkdocs check. After writing docstrings,
  also updates docs/src/API/ pages so every exported symbol appears exactly once,
  organised into logical categories, with stale entries removed.
  Invoke this skill whenever the user mentions: docstring, missing doc,
  undocumented symbol, API doc, checkdocs warning, docs/src/API, @docs block,
  or asks to document a type or function, sync the API pages, or keep the API
  index up to date. Also use it when the user asks to "write docs for" or "add
  docs to" source files, or when a CI failure mentions missing or incomplete
  docstrings.
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

Use an Explore subagent to read 2–3 symbols that already have complete docstrings
to calibrate style. This avoids consuming the main context window with large file
reads. Ask the Explore agent to return the verbatim docstring text for each symbol.

Identify:

- Whether DocStringExtensions macros are used, and which ones (`$(TYPEDEF)`,
  `$(TYPEDFIELDS)`, `$(TYPEDSIGNATURES)`, `$(METHODLIST)`).
- How prose is structured relative to macro-generated content (e.g. does prose
  come before or after `$(TYPEDFIELDS)`?).
- What field documentation pattern is preferred: inline string literals above
  each struct field vs. a separate prose block.
- Which format is used for struct docstrings: the **old format** (an indented
  type-name header on the first line, no `$(TYPEDEF)`, manual `# Constructor`
  section), or the **new format** (prose only, `$(TYPEDEF)` for the signature,
  `$(METHODLIST)` for constructors). Normalise old-format structs to new-format
  during Step 3.

This detected baseline becomes the style target for every new or normalised
docstring. Do not impose a different convention — match what is already there.

### Step 2 — Enumerate candidates

Discover the package name from `Project.toml` (the `name =` field). Then run:

```
grep -nE '^(function |struct |abstract type |mutable struct |const )' src/**/*.jl
```

**Cross-file exports**: exported names may be declared in a central module file
(e.g. `src/PackageName.jl`) while the definition lives in a different file.
Read the module file for all `export` statements so you catch every public symbol
regardless of where it is defined.

For each exported symbol, check whether a non-empty docstring immediately precedes
the definition. Produce a prioritised list:

1. **Missing entirely** — no docstring at all.
2. **Old-format struct** — indented type name on first line, no `$(TYPEDEF)`,
   or redundant manual `# Constructor` / `# Constructors` section alongside
   `$(METHODLIST)`.
3. **Empty or stub** — only a bare macro line (e.g. `$(TYPEDSIGNATURES)`) with
   no prose.
4. **Incomplete** — prose present but key sections absent (missing `# Arguments`,
   `# Examples`, or field strings not describing semantic role).

**Also scan every function in the file for old-style docstrings**, regardless of
whether it is exported. An old-style docstring is one that uses an indented
function-name header (e.g. `    my_func(arg1, arg2)`) and/or an `Args:` /
`Arguments:` block with the `` `name` - description `` format. Convert these to
the `$(TYPEDSIGNATURES)` convention in the same editing pass — the whole file
should end up stylistically uniform.

### Step 3 — Draft docstrings

For each candidate, write a docstring that matches the detected convention.

#### Getters and simple accessors

Simple getters for exported process/struct types (e.g.
`get_prior_mean(p::MyProcess)`) are public API and must be documented if
exported. A one-line `$(TYPEDSIGNATURES)` + short prose sentence suffices; no
`# Arguments` or `# Examples` is needed unless the semantics are non-obvious.

#### Old-format struct docstrings

If a struct docstring starts with an indented type name (e.g. `    MyStruct{...}`),
convert it to the new format:

- Remove the indented type name from the docstring.
- Add `$(TYPEDEF)` immediately after the opening prose sentence.
- Replace any manual `# Constructor` or `# Constructors` section (listing
  function signatures) with a `# Constructors` section containing only
  `$(METHODLIST)`. If `$(METHODLIST)` is already present alongside the manual
  list, remove the manual list.
- Preserve any genuine prose that was in the old `# Constructor` section if it
  explains non-obvious behaviour; discard boilerplate signature repetition.

#### Named constructors (factory functions)

`$(METHODLIST)` only lists methods whose name matches the struct type. Exported functions
that **build an instance of the struct but carry a different name** — factory functions such
as `constrained_gaussian` for `ParameterDistribution` — are invisible to `$(METHODLIST)`
and must be surfaced manually in the struct docstring.

When such functions exist, add a prose note inside the `# Constructors` section,
immediately before `$(METHODLIST)`:

```julia
"""
Structure to hold a parameter distribution, always stored as an array of
distributions internally.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

Recommended construction (for most problems) is via the `constrained_gaussian()`
utility (see its own docstring for details and a usage example).

$(METHODLIST)
"""
struct ParameterDistribution
    ...
end
```

The note should:
- Name the function in backticks.
- Give a one-line hint about when to prefer it over the direct constructor.
- Optionally include a minimal usage snippet if the factory is the primary entry point
  and no separate `# Examples` block exists on the factory function itself.

The factory function still needs its **own full docstring** (`$(TYPEDSIGNATURES)`,
`# Arguments`, `# Examples` if non-trivial). The struct-level note is a pointer,
not a replacement.

**Detecting named constructors during Step 2:** when enumerating candidates, flag
exported functions whose name differs from any type name but whose body or doc
clearly returns an instance of a specific struct. Common signals: the function name
ends with a domain term (e.g. `constrained_gaussian`, `from_file`), its return
statement calls the struct constructor directly, or the existing codebase already
mentions the relationship somewhere in prose.

#### Multiple dispatch — one docstring per concept

When a function has multiple dispatch methods, document **only** the primary
user-facing overload and leave all other overloads undocumented. Competing
docstrings fragment the rendered API docs and create maintenance burden.

The primary overload is the method whose argument type is the **broadest
user-facing type** — e.g. `ParameterDistribution` rather than `Parameterized`
or `Samples`.

**Type-parameter specialisations count as overloads.** If the same function name
is defined for `where {FT, P <: MyProcess{FT, TypeA}}` and
`where {FT, P <: MyProcess{FT, TypeB}}`, these are two dispatch methods of the
same concept. Document only one — typically the first defined, or the more
general one — and leave the rest undocumented.

`update_ensemble!` is a specialised internal update hook called by the framework,
not by users directly. It must **not** be documented even if it is exported.

#### Old-style function docstrings (all functions, not just exported)

Convert any docstring that uses an indented function-name header or an `Args:` /
`Arguments:` section to the `$(TYPEDSIGNATURES)` style, even for internal
(non-exported) helpers. The canonical old-style markers are:

- First line indented with spaces: `    my_func(arg, ...)` — replace with `$(TYPEDSIGNATURES)`.
- Argument block labelled `Args:` or `Arguments:` with `` `name` - description `` lines — replace
  with a `# Arguments` section using `` - `name`: description `` format.

Doing this in the same pass keeps the file stylistically uniform and prevents
old-style docstrings from persisting as invisible technical debt.

#### General rules

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

### Step 4 — Apply edits

When editing files that contain non-ASCII characters (e.g. author names with
accented letters like "Garbuno-Iñigo" or "Nüsken"), the file may store
characters in Unicode NFD form while the Edit tool normalises to NFC, causing
match failures. If an Edit call fails with a "not found" error on a string you
can see in the file, use a Python one-liner to apply the replacement with
NFD-normalised strings:

```bash
python3 - <<'EOF'
import unicodedata, pathlib
p = pathlib.Path("src/MyFile.jl")
text = p.read_text()
old = unicodedata.normalize('NFD', "the old string here")
new = unicodedata.normalize('NFD', "the new string here")
p.write_text(text.replace(old, new, 1))
EOF
```

After a Python edit, re-read the file before making any further Edit calls to
the same file (the Edit tool tracks file state from the last Read).

### Step 5 — Sync `docs/src/API/` pages

After all source-file edits are applied, update the Documenter.jl API pages so
that every exported, documented symbol appears exactly once, organised into
logical categories. The goal is that a reader browsing `docs/src/API/` sees a
complete, non-redundant index of the public API — nothing missing, nothing
stale.

#### 5a — Build the source-to-page map

Read `docs/make.jl` and extract the `api` array to see which display name maps
to which page path (e.g. `"Inversion" => "API/Inversion.md"`). For each page,
read its `@meta` block to find `CurrentModule = ...`. This tells you which
module's exports the page is responsible for.

#### 5b — Collect current `@docs` entries per page

For each API page, extract every symbol entry listed inside ` ```@docs ``` `
blocks. Some entries carry type-signature qualifiers (e.g.
`get_obs(ekp::EnsembleKalmanProcess)`) — track both the raw entry string and
the base name (everything before the first `(`).

#### 5c — Find missing and stale entries

**Exported but not defined (phantom exports).** Before anything else, check
that every exported name actually resolves to a definition — a function, type,
or constant — somewhere in the source files of that module. If an exported name
has no definition anywhere, it is a phantom export: remove the `export`
statement (or just that name from a multi-name `export` line) from the source
file. Do not add phantom exports to any API page.

**Missing from the API.** A symbol is **missing** from a page when it is
exported from that page's `CurrentModule`, its base name does not appear in any
`@docs` block on any API page, and it has a definition in the source. If it
lacks a docstring, go back and write one now (following the conventions from
Steps 1–3) before adding it to the API page — an undocumented entry in a
`@docs` block will cause the docs build to error. Every exported, defined
symbol must end up with a docstring and an API page entry.

**Stale API entries.** An entry is **stale** when the base name is no longer
exported from the module, or the symbol no longer has a definition in the
source.

Run all three checks before making edits so you can see the full diff in one
pass.

#### 5d — Place missing symbols into appropriate sections

Insert each missing symbol into the section of its API page that best matches
its role. Use the existing section headings on the page as the primary guide —
`## Getter functions`, `## Error metrics`, etc. are already established
categories; add the new symbol to the most thematically fitting one.

When no existing section fits, create a new `##` heading that names the
functional group (e.g. `## Accelerators`, `## Utility functions`) and open a
fresh ` ```@docs ``` ` block below it. Avoid catch-all sections like
`## Miscellaneous`; if you find yourself reaching for that, split more finely.

Broad heuristics for categorisation when the page has no existing sections to
guide you:

- Struct / abstract type → primary types section (first block on page)
- Functions starting with `get_` → `## Getter functions`
- Functions starting with `compute_`, `construct_`, `build_` → a computation
  or construction section
- Update or step functions → an operations section
- Error-metric functions → `## Error metrics`
- Scheduler or controller types/functions → their own named section

For a multiple-dispatch function where only the primary overload is documented
(per Step 3), list only that overload. If the existing page convention uses
type-qualified entries (e.g. `foo(x::MyType)`), follow that convention;
otherwise use the plain name.

#### 5e — Remove stale entries

For each stale API entry:

1. Delete the line from its `@docs` block. If that empties the block, delete
   the block. If that empties the section, delete the section heading too.
2. If the symbol is stale because it is no longer defined (phantom export),
   also remove the `export` statement from the source file. For multi-name
   export lines (e.g. `export foo, bar, baz`), remove only the stale name and
   leave the rest intact.

#### 5f — Ensure no symbol appears on two pages

Each base name must appear on at most one API page. If you find a duplicate,
keep it on the page whose `CurrentModule` matches the module where the symbol
is defined, and remove it from the other page.

### Step 6 — Verify

Find the package name from `Project.toml`, then confirm the package loads
without error:

```
julia --project -e 'import Pkg; Pkg.instantiate(); using <PackageName>'
```

If a docs build is configured (`docs/make.jl` is present), run it and resolve
any `checkdocs` warnings introduced by the new docstrings.

### Step 7 — Offer to improve the skill

Once the docs build is clean, ask the user: "Would you like to improve the
**docstrings** skill itself using skill-creator? You can share suggestions, or I
can analyse patterns from this session — recurring edge cases, formatting
decisions, or anything that felt awkward — to refine the skill for next time."

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
- **Multiple-dispatch — one docstring per concept**: Document only the primary
  user-facing overload (the method taking the top-level composite type). All
  other dispatch methods remain undocumented. Do **not** add `$(METHODLIST)` to
  function docstrings — `$(TYPEDSIGNATURES)` already surfaces all overloads.
  `$(METHODLIST)` belongs only in struct docstrings (inside `# Constructors`).
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
| **Convention parity** | High | New docstrings use the same macro set and structural pattern as the best-documented symbols already present. Old-format struct docstrings have been normalised. |
| **Informativeness** | Medium | Prose answers "what, when, why". Units present for physical quantities. `# Arguments` section present where needed. `# Examples` jldoctest block present for non-trivial public functions. |
| **No duplication** | Medium | Prose does not duplicate macro-generated content. Field string literals do not restate the field's type. No redundant manual `# Constructor` section alongside `$(METHODLIST)`. |
| **API page coverage** | High | Every exported, documented symbol appears exactly once across `docs/src/API/` pages. No stale entries. Symbols are grouped into descriptive sections. |
| **Correctness** | High | Package loads without error; docs build (if configured) completes without new warnings. |

## Examples

### Struct: old format → new format

```julia
## Before — OLD format: indented type name, no $(TYPEDEF), manual # Constructor section

"""
    Sampler{FT<:AbstractFloat, T <:SamplerType} <: Process

An ensemble Kalman Sampler process. with type Sampler Type (e.g., ALDI or EKS).

# Constructor
Sampler(prior::ParameterDistribution) # ALDI update (sampler_type="aldi")
Sampler(prior::ParameterDistribution; sampler_type = "eks") # EKS update

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Sampler{FT <: AbstractFloat, T} <: Process
    ...
end

## After — NEW format: prose only, $(TYPEDEF) for signature, $(METHODLIST) for constructors

"""
An ensemble Kalman Sampler process parameterised by algorithm type `T <: SamplerType` (`ALDI` or `EKS`).

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Sampler{FT <: AbstractFloat, T} <: Process
    ...
end
```

### Function: stub docstring improved

```julia
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

### Multiple-dispatch: type-parameter specialisations

```julia
## Before — both specialisations documented (anti-pattern)

"""
$(TYPEDSIGNATURES)

Returns the updated parameters using the EKS algorithm.
"""
function eks_update(ekp, u, g, process::PEKS) where {FT, PEKS <: Sampler{FT, EKS}}
    ...
end

"""
$(TYPEDSIGNATURES)

Returns the updated parameters using the ALDI algorithm.
"""
function eks_update(ekp, u, g, process::PALDI) where {FT, PALDI <: Sampler{FT, ALDI}}
    ...
end

## After — only the first overload documented; second left bare

"""
$(TYPEDSIGNATURES)

Return the updated parameter vectors using the EKS sampler algorithm
(Garbuno-Iñigo, Hoffmann, Li, Stuart 2019).
"""
function eks_update(ekp, u, g, process::PEKS) where {FT, PEKS <: Sampler{FT, EKS}}
    ...
end

function eks_update(ekp, u, g, process::PALDI) where {FT, PALDI <: Sampler{FT, ALDI}}
    ...
end
```

### Simple getter: minimal docstring

```julia
## Before — getter with no docstring

get_prior_mean(process::Sampler) = process.prior_mean

## After — one-liner is enough

"""
$(TYPEDSIGNATURES)

Return the prior mean vector stored in `process`.
"""
get_prior_mean(process::Sampler) = process.prior_mean
```
