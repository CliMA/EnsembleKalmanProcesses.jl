# AI Specialisms

This directory contains prompt-engineering specifications ("specialisms") that guide Claude on recurring task types in this repository. Each specialism is a self-contained XML file bundling a task definition, step-by-step workflow, quality rubric, and formatting rules.

A `UserPromptSubmit` hook in `.claude/settings.json` injects this file into every session. Claude must identify the matching specialism before acting, or ask the user for direction if none clearly fits.

## Specialism index

| Name | File | Summary | Trigger keywords |
|------|------|---------|-----------------|
| docstrings | [docstrings.xml](docstrings.xml) | Add or normalize Julia docstrings on public symbols using the package's established convention | docstring, missing doc, undocumented, API doc |
| docs-neaten | [docs-neaten.xml](docs-neaten.xml) | Cross-cutting style pass on any docs page: prose, `@ref` correctness, code-fence tags, heading hierarchy, link form | neaten, polish, prose, cleanup, consistency, cross-reference, formatting, refs |
| docs-readme | [docs-readme.xml](docs-readme.xml) | Write or improve the project-root README.md: badge table, pitch, copy-paste demo, absolute URLs | readme, README, github landing, badges |
| docs-index | [docs-index.xml](docs-index.xml) | Write or improve the docs site landing page: package pitch, `@ref` cross-refs, routing hub | index page, landing page, docs home, docs landing |
| docs-installation | [docs-installation.xml](docs-installation.xml) | Write or improve an installation or setup page: imperative procedures, `!!! info` admonitions, no cross-refs | install, installation, setup, getting started |
| docs-tutorial | [docs-tutorial.xml](docs-tutorial.xml) | Write or improve a hand-written tutorial page in `docs/src/examples/` only (not Literate-generated `docs/src/literated/`): alternating prose-and-code, first-person plural narration, double-backtick math | tutorial, example, walkthrough, worked example |
| base-show | [base-show.xml](base-show.xml) | Add concise `Base.show` methods for types with unhelpful default REPL output | show, display, print, repr, REPL output |
| api-page | [api-page.xml](api-page.xml) | Write or update a hand-curated `@docs`-block API reference page in `docs/src/API/` | api page, @docs block, exported symbols, API reference |
| stale-cleanup | [stale-cleanup.xml](stale-cleanup.xml) | Detect and remove stale editor backups, orphan tmp files, and coverage artifacts from the working tree | stale files, backup files, \*~ files, tmp, cov, cleanup |

## Adding a new specialism

1. Copy `_template.xml` → `ai/<name>.xml`.
2. Fill in all four sections (`<task>`, `<workflow>`, `<rubric>`, `<formatting>`).
3. Add a row to the index table above.
4. Keep steps general: describe how to *discover* targets (grep, module introspection) rather than naming specific files or symbols.

## How activation works

On every user prompt, the hook runs:

```sh
cat ai/README.md && printf '\n---\nSPECIALISM ROUTING ...\n'
```

Claude receives this index plus the routing instruction. It must:

- If a specialism clearly matches → read the full XML file, then follow its task/workflow/rubric/formatting.
- If no specialism clearly matches → pause and ask the user to pick an existing specialism or approve a new one before doing any work.

## Authoring principle

Specialisms describe a *class* of task, not a specific instance. Workflow steps say "find candidates by grepping for struct declarations" — not "edit `src/Foo.jl`". This makes every specialism valid as the codebase grows and refactors.
