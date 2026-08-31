# Lorenz examples

This directory contains several self-contained examples based on the Lorenz dynamical systems.

## Lorenz 63

A detailed Lorenz 63 example, split into two parts:
- [`explore_lorenz_63.jl`](explore_lorenz_63.jl) explores the dynamical system itself: sensitivity to initial conditions vs. parameter perturbations, statistics windows, and the calibration loss landscape.
- [`Lorenz_63_example.jl`](Lorenz_63_example.jl) calibrates the Lorenz 63 parameters `(ρ, β)` with EKI, and compares against a derivative-based Levenberg-Marquardt solve.

[`Lorenz_63_notebook.jl`](Lorenz_63_notebook.jl) and [`explore_lorenz_63_notebook.jl`](explore_lorenz_63_notebook.jl) contain the same two scripts refactored into notebook-friendly (`# %%` cell-delimited) form, with their plotting/setup helpers split out into companion `Lorenz_63_notebook_utils.jl` / `explore_lorenz_63_notebook_utils.jl` files. These are linked to Google Colab notebooks for interactive use, with no local Julia installation required:
- Explore: https://colab.research.google.com/drive/1H_u2g9rLkD5fMJkVdEa6w-W4691msITO?usp=sharing
- Calibrate: https://colab.research.google.com/drive/12obyLGzS1poWkUuvuw4A7s5RSUsMT5xA?usp=sharing

## Lorenz 96

Two examples calibrating the Lorenz 96 forcing parameter(s) with EKI:
- [`Lorenz_96_example.jl`](Lorenz_96_example.jl) learns a single, spatially-constant forcing term.
- [`Lorenz_96_example_spatial_dep_forcing.jl`](Lorenz_96_example_spatial_dep_forcing.jl) learns a spatially-varying forcing term.

## Parallelization

[`distributed_Lorenz_96_example.jl`](distributed_Lorenz_96_example.jl) demonstrates several ways to parallelize the forward map evaluations across ensemble members (multithreading, `pmap`, and `@distributed for`), as described in [`../../docs/src/parallel_hpc.md`](../../docs/src/parallel_hpc.md).
