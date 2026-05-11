---
name: docs-installation
description: >
  Write or improve an installation or getting-started documentation page for a Julia package.
  Use this skill whenever the user mentions: installation docs, setup docs, getting started page,
  install instructions, developer setup, or any task that involves writing or fixing a page that
  walks a reader through installing a Julia package or setting up a development environment.
  Also trigger when the user asks to "improve the installation page", "add developer setup steps",
  "fix the Pkg.add command in the docs", or anything that touches how-to-install content in docs/.
---

Write or improve a setup or installation page (for example `docs/src/installation.md` or
`docs/src/getting_started.md`). The page walks a reader from zero to a working Julia
environment with the package installed, and optionally covers developer workflows (cloning
the repository, running the test suite, building the docs locally). It is almost entirely
procedural: every step should be actionable, accurate, and verifiable by the reader without
consulting another page.

## Workflow

### Step 1 — Gather project facts

Read the following files to collect the facts the page depends on:

- `Project.toml` — the registered package name (used in `Pkg.add`) and Julia compat bounds.
- `test/runtests.jl` — whether a test suite exists and what command invokes it.
- `docs/make.jl` — whether a docs build exists and what command runs it.

These facts determine which procedures to include and what version constraints to state.

### Step 2 — Locate the installation page

Run:

```sh
find docs/src -maxdepth 2 \( -name '*install*' -o -name '*getting_start*' \)
```

Read the current page if one exists. Identify: missing procedures, inaccurate commands,
untagged code fences, prose tips that belong in admonitions, and a missing H1.

### Step 3 — Draft or edit the page

Follow this canonical structure:

1. **H1** — "Installation" or "Getting started" (match the existing project convention).
2. **Opening sentence** — state that the package is a registered Julia package (if true).
3. **H3 for each distinct procedure** — skip H2 entirely; the convention for installation
   pages is H1 then H3 for all sub-sections. Typical procedures:
   - H3 Basic installation (`Pkg.add`)
   - H3 Cloning the repository (for contributors)
   - H3 Running the test suite
   - H3 Building the documentation locally
4. **Optional steps and alternative paths** — place these inside `!!! info` admonitions,
   not as inline paragraphs.

The page must be self-contained: avoid `@ref` cross-references. If you need to link to
another docs page (e.g., a tutorial), use a relative Markdown path.

### Step 4 — Verify accuracy

Check that:

- The `Pkg.add` package name exactly matches the `name` field in `Project.toml`.
- Julia version constraints are consistent with the `[compat]` section of `Project.toml`.
- Any test or docs build commands match the current project layout.

### Step 5 — Build the docs

Run the docs build to confirm the page renders without new warnings:

```sh
julia --project=docs docs/make.jl
```

Resolve any Documenter warnings introduced by your edits before reporting the task complete.

### Step 6 — Offer to improve the skill

Once the docs build is clean, ask the user: "Would you like to improve the **docs-installation** skill itself using skill-creator? You can suggest additions to the workflow or quality criteria, or I can use what came up during this session — missing steps, awkward conventions, project-specific wrinkles — to make the skill better for next time."

## Quality criteria

- **Imperative voice** (high priority): All instructions use second-person imperative —
  "install the package", "run the following", "press ]". Never use passive voice
  ("the package can be installed by...") or third-person.
- **Accurate snippets** (high priority): `Pkg.add` uses the exact registered package name
  from `Project.toml`. Julia version bounds match the compat section. Test and build
  commands match the current project layout.
- **H3-only sub-sections** (medium): Sub-sections are H3. The H1 → H3 skip (no H2) is
  intentional for installation pages and should be preserved even though it departs from
  the general heading-hierarchy convention.
- **Admonitions for asides** (medium): Optional tips, alternative paths, and gotchas
  appear in `!!! info "Friendly title"` blocks, not as inline paragraphs. Use
  `!!! warning "Title"` for real gotchas (e.g., known version incompatibilities).
- **Self-contained** (medium): The page does not rely on `@ref` cross-references. A reader
  who arrives directly on this page via a search engine can complete the installation
  without navigating elsewhere.

## Formatting rules

- H1 as the page title; all sub-sections as H3 (skipping H2 is intentional).
- Julia Pkg-mode commands: show the `]` prompt on its own line, then the `pkg>` command.
  Alternatively, use the functional form inside a `julia` block.

  ```julia
  # Pkg REPL style
  julia> ]
  pkg> add MyPackage

  # Functional style
  using Pkg
  Pkg.add("MyPackage")
  ```

- Shell commands go in a `sh` fenced block; Julia commands in a `julia` fenced block.
  Never leave a code fence untagged.
- `!!! info "Friendly title"` for optional tips and alternative paths.
  `!!! warning "Title"` for real gotchas (e.g., version incompatibilities).
- No `@ref` links. If pointing to another docs page, use a relative Markdown path.
- No badges, embedded images, or figures.
- Second-person throughout ("you", "your"). No first-person plural ("we").

## Before / after example

**Before** — missing H1, untagged fence, version note buried in prose:

    MyPackage is a registered Julia package. To install:

        using Pkg; Pkg.add("MyPackage")

    If you want to run the tests you can do `Pkg.test("MyPackage")`.
    Note: you need Julia 1.9 or higher.

**After** — H1, tagged fences, H3 sub-sections, `!!! info` admonition:

    # Installation

    MyPackage is a registered Julia package. Install it from the Julia package manager:

    ```julia
    using Pkg
    Pkg.add("MyPackage")
    ```

    !!! info "Julia version requirement"
        MyPackage requires Julia 1.9 or higher. Check your version with `julia --version`.

    ### Running the test suite

    To verify your installation, run the package tests:

    ```julia
    using Pkg
    Pkg.test("MyPackage")
    ```

    ### Building the documentation locally

    ```sh
    julia --project=docs docs/make.jl
    ```
