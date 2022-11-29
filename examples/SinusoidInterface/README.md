## Running this example.

First instantiate the example,
```
cd examples/SinusoidInterface
julia --project
]
instantiate
```
Then run the bash script
```
bash calibrate_script
```
The prior distributions are stored in `parameters.toml` in TOML format. More examples defining parameter distributions in this way are given in `test/TOMLInterface/toml/`.

After the example runs, (it takes *several minutes* to run, see below...), results are stored in `output/eki.jld2`. To inspect the results, load up with `julia --project` as follows
```julia
julia> using EnsembleKalmanProcesses, JLD2

julia> @load "output/eki.jld2"
3-element Vector{Symbol}:
 :eki
 :param_dict
 :prior
```
Then, for example the final 5-member parameter ensemble is given by:
```julia
julia> get_ϕ_final(prior,eki)
2×5 Matrix{Float64}:
 1.31749  1.28875  1.27677  1.33311  1.3305
 6.27765  6.63572  6.65148  6.66096  6.56518
```

## Why run this example in this way?

The purpose of this example is to demonstrate how the EKP update and the computer model are **completely independent** (e.g. the model can be written in another language, can require HPC and any data-processing tools). This calibration is called from a simple script-language (bash) and the interactions between the calibration tools and the forward map are only through simple readable files.

- When ensembles are created, the ensemble member parameters are stored in separate directories, each containing a file in a universal TOML format
`output/iteration_X/member_Y/parameters.toml`
- Each `run_computer_code.jl` reads from one specific parameter file given from the pair `(X,Y)`, and outputs data into this directory (stored in JLD2, but could be any format)
`output/iteration_X/member_Y/model_output.jld2`
- Each `update_EKP.jl` call reads all ensemble member outputs from the JLD2 files, collects them together, updates the ensemble, and writes the new parameter ensemble to file.

## Why is it so slow?
This example is for instructional purposes only. It is slow for two reasons,
1. The forward map is written in Julia, and so 99.99% of computation time is precompilation. This can be accelerated by using compiled system-images, e.g `PackageCompiler.jl`. System images are not reproducible, therefore we have not implemented this.
2. We do not parallelize forward map evaluation over ensemble members. (We do not want this example to depend on workload managers.) Please see our docs pages for information about parallelization.
