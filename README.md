# EnsembleKalmanProcesses.jl
Implements optimization and approximate uncertainty quantification algorithms, Ensemble Kalman Inversion, and Ensemble Kalman Processes.


| **Documentation**    | [![dev][docs-latest-img]][docs-latest-url]       |
|----------------------|--------------------------------------------------|
| **DOI**              | [![DOI][zenodo-img]][zenodo-latest-url]          |
| **Docs Build**       | [![docs build][docs-bld-img]][docs-bld-url]      |
| **Unit tests**       | [![unit tests][unit-tests-img]][unit-tests-url]  |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]           |
| **JOSS**             | [![status][joss-img]][joss-url]                  |

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.6382967.svg
[zenodo-latest-url]: https://doi.org/10.5281/zenodo.6382967

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://CliMA.github.io/EnsembleKalmanProcesses.jl/dev/

[docs-bld-img]: https://github.com/CliMA/EnsembleKalmanProcesses.jl/actions/workflows/Docs.yml/badge.svg?branch=main
[docs-bld-url]: https://github.com/CliMA/EnsembleKalmanProcesses.jl/actions/workflows/Docs.yml

[unit-tests-img]: https://github.com/CliMA/EnsembleKalmanProcesses.jl/actions/workflows/Tests.yml/badge.svg?branch=main
[unit-tests-url]: https://github.com/CliMA/EnsembleKalmanProcesses.jl/actions/workflows/Tests.yml

[codecov-img]: https://codecov.io/gh/CliMA/EnsembleKalmanProcesses.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/EnsembleKalmanProcesses.jl

[joss-img]: https://joss.theoj.org/papers/5cb2d4c6af8840af61b44071ae1e672a/status.svg
[joss-url]: https://joss.theoj.org/papers/5cb2d4c6af8840af61b44071ae1e672a

### Requirements
Julia LTS version or newer

## What does the package do?
EnsembleKalmanProcesses (EKP) enables users to find an (locally-) optimal parameter set `u` for a computer code `G` to fit some (noisy) observational data `y`. It uses a suite of methods from the Ensemble Kalman filtering literature that have a long history of success in the weather forecasting community.

What makes EKP different?
- EKP algorithms are efficient, and able to optimize high-dimensional parameters over very noisy cost landscapes.
- We don't require differentiating the model `G` at all! you just need to be able to run it at different parameter configurations.
- We don't even require `G` to be in Julia!
- Ensemble model evaluations are fully parallelizable - so we can exploit our HPC systems capabilities!
- We provide some lego-like interfaces for creating complex priors and observations.
- We provied easy interfaces to toggle between many different algorithms and configurable features.

## What does it look like to use?

Below we will outline the current user experience for using `EnsembleKalmanProcesses.jl` to solve the classic inverse problem where we learn `y = G(u) + e`, for `e` distributed with `N(0,Γ)`. Given some prior knowledge of the parameters `u` in the problem (say we have five, one positive and four more that are more uncertain) we are ready to go! 

```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

prior_u1 = constrained_gaussian("positive_with_mean_2", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("four_with_spread_5", 0, 5, -Inf, Inf, repeats=4)
prior = combine_distributions([prior_u1, prior_u2]) 
using Plots
plot(prior) # lets see it with Plots.jl

N_ensemble = 10 # ten ensemble members
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
ensemble_kalman_process = EnsembleKalmanProcess(
    initial_ensemble, 
    y, 
    Γ, 
    Inversion() # use Ensemble Kalman Inversion updates
)

N_iterations = 5
for i in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat(
        [G(params_i[:, i]) for i in 1:N_ensemble]... # I'm easy to parallelize!
    )

    update_ensemble!(ensemble_kalman_process, G_ens)
end

final_solution = get_ϕ_mean_final(prior, ensemble_kalman_process) 
```
With suitable parallelization, the final solution is obtained in the time it takes to perform ~5 iterations, `G` and five EKP update steps.

See this example working [here!](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/literated/sinusoid_example/). check out our many example scripts above in `examples/`

# [Quick links!](@id quick-links)

- [How do I build prior distributions?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/)
- [How do I build my observations and encode batching?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/)
- [What ensemble size should I take? Which process should I use? What is the recommended configuration?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/defaults/)
- [What is the difference between `get_u` and `get_ϕ`? Why do the stored parameters apperar to be outside their bounds?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/)
- [What can be parallelized? How do I do it in Julia?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parallel_hpc/)
- [What is going on in my own code?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/troubleshooting/)
- [What is this error/warning/message?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/troubleshooting/)
- [Where can I walk through a simple example?](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/literated/sinusoid_example/)


## Citing us

If you use the examples or code, please cite [our article at JOSS](https://joss.theoj.org/papers/10.21105/joss.04869) in your published materials.


### Getting Started 
![eki-getting-started](https://github.com/CliMA/EnsembleKalmanProcesses.jl/assets/45243236/e083ab8c-4f93-432f-9ad5-97aff22764ad)
<!---
# Link to Miro for editing photo (ask haakon for access): https://miro.com/app/board/uXjVNm_1teY=/?share_link_id=329380184889  
-->
