# Installation

EnsembleKalmanProcesses.jl is a registered Julia package. You can install the latest version
of EnsembleKalmanProcesses.jl through the built-in package manager. Press `]` in the Julia REPL
command prompt and

```julia
julia> ]
(v1.5) pkg> add EnsembleKalmanProcesses
(v1.5) pkg> instantiate
```

This will install the latest tagged release of the package.

### Cloning the repository

Alternatively, you can clone the repository and then then instantiate:

```
> cd EnsembleKalmanProcesses.jl
> julia --project -e 'using Pkg; Pkg.instantiate()'
```

To test that the package is working:

```
> julia --project -e 'using Pkg; Pkg.test()'
```

### Building the documentation locally

Once the project is built, you can build the project documentation under the `docs/` sub-project:

```
> julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
> julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`