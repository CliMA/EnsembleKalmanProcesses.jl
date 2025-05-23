# [Parallelism and High Performance Computing (HPC)](@id parallel-hpc)

One benefit of ensemble methods is their ability to be parallelized. The parallellism occurs outside of the EKP update, and so is not present in the source code. On this page we provide suggestions for parallelization for two types of problems:
1. Run a parallel loop or map within your current Julia session.
2. Run a parallel loop through a read/write file interface and workload manager. We provide some utilities to read/write files in a TOML format.

## Case 1: Parallel code within Julia
Let's look at the simple example for the [Lorenz 96 dynamical system](@ref Lorenz-example). In particular we'll focus on the evaluations of the Lorenz 96 solver explicitly written in [`GModel.jl`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/Lorenz/GModel.jl) that contains the following loop over the `N_ens`-sized ensemble:

```julia
function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = zeros(nd, N_ens)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[:, i] = lorenz_forward(settings, lorenz_params[i])
    end
    return g_ens
end
```
Each ensemble member `i` runs the Lorenz 96 model with `settings` configuration, and model parameters `lorenz_params[i]`. The runs do not interact with each other, and the user has several options to parallelize.

## Running examples:
All of the following example cases are covered in [`distributed_Lorenz_example.jl`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/Lorenz/distributed_Lorenz_example.jl). At the top of file uncomment one of the following options
1. `case = multithread` 
2. `case = pmap` 
3. `case = distfor` 

### Multithreading, `@threads`

To parallelize with multithreading, julia must call the file with a prespecified number of threads. For example, for 4 threads, 
```
$ julia --project -t 4 distributed_Lorenz_example.jl
```
We exploit the multithreading over `N_ens` ensemble members in this example with the following loop in [`GModel_multithread.jl`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/Lorenz/GModel_multithread.jl):
```julia
function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = zeros(nd, N_ens)
    Threads.@threads for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[:, i] = lorenz_forward(settings, lorenz_params[i])
    end
    return g_ens
end
```
You can read more about multi-threading [here](https://docs.julialang.org/en/v1/manual/multi-threading/).

### Parallel map, `pmap`
When using multiple processes, the Julia environment must first be loaded on each worker processor. We include these lines in the main file
```julia
using Distributed
addprocs(4; exeflags = "--project")
```
And we would call the file is called by
```
$ julia --project distributed_Lorenz_example
```
This ensures that we obtain `4` worker processes that are loaded with julia's current environment specified by `--project` (unlike when calling `julia --project -p 4`). We use  `pmap` to apply a function to each element of the list (i.e the ensemble member configurations). For example, see the following code from [`GModel_pmap.jl`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/Lorenz/GModel_pmap.jl),
```julia
using Distributed
function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = zeros(nd, N_ens)
    g_ens[:, :] = vcat(pmap(x -> lorenz_forward(settings, x), lorenz_params)...)
    return g_ens
end
```
If `pmap` is called within a module, that module will also need to be loaded on all workers. For this we use the macro `@everywhere module XYZ`.

You can read more about `pmap` [here](https://docs.julialang.org/en/v1/manual/distributed-computing/#Parallel-Map-and-Loops).

### Distributed loop, `@distributed for` 
When using multiple processes, the Julia environment must also be loaded on each worker processor. We include these lines in the main file
```julia
using Distributed
addprocs(4; exeflags = "--project")
```
And we would call the file is called by
```
$ julia --project distributed_Lorenz_example
```
When using distributed loops, it is necessary to be able to write to shared memory. To do this we use the [SharedArrays](https://docs.julialang.org/en/v1/manual/distributed-computing/#man-shared-arrays) package. For example, see the following distributed loop in [`GModel_distfor`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/Lorenz/GModel_distfor.jl) 
```julia
using Distributed
using SharedArrays
function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = SharedArray{Float64}(nd, N_ens)
    @sync @distributed for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[:, i] = lorenz_forward(settings, lorenz_params[i])
    end
    return g_ens
end
```
`@sync` forces the code to wait until all processes in the `@distributed for` loop are complete before continuing.

If `@distributed for` is used within another module, that module will also need to be loaded on each worker processor. For this we use the macro `@everywhere module XYZ`.

!!! note
    `@distributed for` is most performant when there is a large ensemble, `N_ens`, and the forward map is computationally cheap. Otherwise, `pmap` is usually the preferred choice.
You can read more about `@distributed` [here](https://docs.julialang.org/en/v1/manual/distributed-computing/#Parallel-Map-and-Loops)

### Case 2: HPC interface

Some applications involve interfacing with non-Julia code or using HPC workload managers. In these cases we suggest using an alternative workflow where one interleaves scripts that launch EKP updates and scripts that runs the model. One possible implementation is the following loop
- Step `0`(a). Write an ensemble of parameter files `parameters_0_i` for `i = 1:N_ens`, with each parameter file containing a sample from the prior distribution.
- Step `0`(b). Construct EKP EKP object and save in `jld2` format e.g. `ekpobject.jld2`.

- for `n = 1,..., N_it`, do:
  - Step `n`(a). Run `N_ens` forward models, with forward model `i` running with the corresponding parameter file `parameters_{n-1}_i`. Write processed output data to file `data_{n-1}_i`.
  - Step `n`(b). Run the ensemble update, by loading both `ekpobject.jld2` and reading in the parameter files `data_{n-1}_i` for `i = 1:N_ens`. Perform the EKP update step. Write new parameter files `parameters_n_i` for `i = 1:N_ens`. Save the ensemble object in `ekpobject.jld2`
  - iterate `n -> n+1`.

For a simple implementation of this, please see the example in [`examples/SinusoidInterface`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/SinusoidInterface/), which is a runnable reimplementation of our [sinusoid example](@ref sinusoid-example) in such a formulation.

In [HPC interfacing example: ClimateMachine](@ref) we implement a similar loop to interface with a SLURM workload manager for HPC. Here, `sbatch` scripts are used to run each component of the calibration procedure. The outer loop over the EKP iterations lives in the overarching `sbatch` script, and for each iteration, the inner loop are realised as "arrays" of slurm jobs (`1, ..., N_ens`), launched for each ensemble member. The code excerpt below, taken from [`ekp_calibration.sbatch`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/examples/ClimateMachine/ekp_calibration.sbatch) for details), illustrates this procedure:
```csh
# First call to calibrate.jl will create the ensemble files from the priors
id_init_ens=$(sbatch --parsable ekp_init_calibration.sbatch)
for it in $(seq 1 1 $n_it)
do
# Parallel runs of forward model
if [ "$it" = "1" ]; then
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_init_ens --array=1-$n ekp_single_cm_run.sbatch $it)
else
    id_ens_array=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_ek_upd --array=1-$n ekp_single_cm_run.sbatch $it)
fi
id_ek_upd=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$id_ens_array --export=n=$n ekp_cont_calibration.sbatch $it)
done
```
Here a dependency tree is set up in SLURM, which iterates calls to the scripts `ekp_single_cm_run.sbatch` (which runs the forward model on HPC) and `ekp_cont_calibration.sbatch` (which performs an EKI update). We find this a smooth workflow that uses HPC resources well, and can likely be set up on other workload managers.

For more details see the [code](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples/ClimateMachine) and docs for the [HPC interfacing example: ClimateMachine](@ref).

