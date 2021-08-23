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

!!! info "But I wanna be on the bleeding edge..."
    If you want the *most recent* developer's version of the package then
    
    ```julia
    julia> ]
    (v1.5) pkg> add EnsembleKalmanProcesses#main
    (v1.5) pkg> instantiate
    ```
    
You can run the tests via the package manager by:

```julia
julia> ]
(v1.5) pkg> test EnsembleKalmanProcesses
```

### Cloning the repository

If you are interested in getting your hands dirty and modifying the code then, you can also
clone the repository and then instantiate, e.g.,

```
> cd EnsembleKalmanProcesses.jl
> julia --project -e 'using Pkg; Pkg.instantiate()'
```

You can run the package's tests:

```
> julia --project -e 'using Pkg; Pkg.test()'
```

!!! info "Do I need to clone the repository?"
    Most times, cloning the repository in not a necessity. If you only wanna use the package's
    functionality then merely adding the packages as a dependency on your project will do the
    job.

### Building the documentation locally

Once the project is built, you can build the project documentation under the `docs/` sub-project:

```
> julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
> julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`