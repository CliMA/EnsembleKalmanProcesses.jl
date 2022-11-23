# TOML interface for fitting parameters of a sinusoid

Here we revisit the [simple sinusoid example](@ref sinusoid-example), with the purpose of demonstrating how the EKP tools can interact with models in a non-intrusive fashion. In particular, this example will be useful for users whose model is written in another language, requires HPC management, or requires additional data processing stages.

We can generally orchestrate the different stages of the calibration process using a script file. Here we employ a Linux bash script:
```
> bash calibrate_script
```
Which executes the following sequence of processes (in this case, calls to `julia --project`)
```
# generate data
julia --project generate_data.jl 

# create initial ensemble and build EKI
julia --project initialize_EKP.jl 

# the EKI loop
for i in $(seq 1 $N_iterations); do

    # run the model at each ensemble member
    for j in $(seq 1 $N_ensemble); do
        julia --project run_computer_model.jl $i $j
    done

    # update the ensemble with EKI
    julia --project update_EKP.jl $i
    
done
```
The interaction between the calibration tools and the forward map are only through simple readable files stored in a nested file structure: for iteration `i`, ensemble member `j`
- Parameter values are stored (with their priors) in `output/iteration_i/member_j/parameters.toml`
- Computer model outputs are stored in `output/iteration_i/member_j/model_output.jld2`

# Inputs and Outputs

The prior distributions are provided in `priors.toml` in TOML format.
```toml
[amplitude]
prior = "Parameterized(Normal(0.5815754049028404, 0.47238072707743883))"
constraint = "bounded_below(0.0)"
description = """
The amplitude of the sine curve.
This yields a physical prior that is log-normal with approximate (mean,sd) = (2,1)
"""

[vert_shift]
prior = "Parameterized(Normal(0,5))"
constraint = "no_constraint()"
description = """
The vertical shift of the sine curve.
This yields a physical prior that is Normal with (mean,sd) = (0,5)
"""
```
More information on the priors and constraints are given [here](@ref parameter-distributions). More examples defining priors in TOML format may be found in `test/TOMLInterface/toml/`.

After running the example, (it takes *several minutes* to run), results are stored in `output/eki.jld2`. To view the results, one can interact with the stored objects by loading `julia --project` and proceeding as follows:
```julia
julia> using EnsembleKalmanProcesses, JLD2

julia> @load "output/eki.jld2"
3-element Vector{Symbol}:
 :eki
 :param_dict
 :prior
```
Then, for example, the final 6-member parameter ensemble is retrieved with:
```julia
julia> get_ϕ_final(prior,eki)
2×6 Matrix{Float64}:
 1.38462  1.36124  1.32444  1.26686  1.33462  1.3636
 6.51158  6.31867  6.68542  6.12809  6.44726  6.52448
```
while the initial ensemble is retrieved with:
```julia
julia> get_ϕ(prior, eki, 1)
2×6 Matrix{Float64}:
  1.05344   1.67949    6.29847  0.951586  2.07678    1.62284
 -7.00616  -0.931872  -6.11603  0.984338  0.274007  -1.39082
```

!!! note "Why is it so slow?"
    The example is slow because the forward map is written in Julia, and so 99.99% of computation for each call to `julia --project` is precompilation. Ensembles in Julia can be accelerated by using methods discussed [here](@ref parallel-hpc), or by compiling system images with `PackageCompiler.jl`, for example. This example is for instructional purposes only, and so is not optimized.