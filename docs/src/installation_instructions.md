# Installation

To build the top-level project, first clone the repository, then instantiate:

```
cd EnsembleKalmanProcesses.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

To test that the package is working:

```
> julia --project -e 'using Pkg; Pkg.test()'
```

### Building the documentation locally

Once the project is built, you can build the project documentation under the `docs/` sub-project:

```
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`