# # Minimization of simple loss functions with sparse EKI
#
# First we load the required packages.

using Distributions, LinearAlgebra, Random, Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
# ## Loss function with single minimum
#
# Here, we minimize the loss function
# ```math
# G₁(u) = \|u - u_*\| ,
# ```
# where ``u`` is a 2-vector of parameters and ``u_*`` is given; here ``u_* = (1, 0)``. 
u★ = [1, 0]
G₁(u) = [sqrt((u[1] - u★[1])^2 + (u[2] - u★[2])^2)]
nothing # hide

# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
Random.seed!(rng_seed)
nothing # hide

# We set a stabilization level, which can aid the algorithm convergence
dim_output = 1
stabilization_level = 1e-3
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output)

# The functional is positive so to minimize it we may set the target to be 0,
G_target = [0]
nothing # hide

# ### Prior distributions
#
# As we work with a Bayesian method, we define a prior. This will behave like an "initial guess"
# for the likely region of parameter space we expect the solution to live in.
prior_u1 = Dict("distribution" => Parameterized(Normal(0, 2)), "constraint" => no_constraint(), "name" => "u1")
prior_u2 = Dict("distribution" => Parameterized(Normal(0, 2)), "constraint" => no_constraint(), "name" => "u2")

prior = ParameterDistribution([prior_u1, prior_u2])

# ### Calibration
#
# We choose the number of ensemble members and the number of iterations of the algorithm
N_ensemble = 20
N_iterations = 10
nothing # hide

# The initial ensemble is constructed by sampling the prior
initial_ensemble = EKP.construct_initial_ensemble(prior, N_ensemble; rng_seed = rng_seed)

# Sparse EKI parameters
γ = 1.0
threshold_value = 1e-2
reg = 1e-3
uc_idx = [1, 2]

process = SparseInversion(γ, threshold_value, uc_idx, reg)

# We then initialize the Ensemble Kalman Process algorithm, with the initial ensemble, the
# target, the stabilization and the process type (for sparse EKI this is `SparseInversion`). 
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, G_target, Γ_stabilization, process)

# Then we calibrate by *(i)* obtaining the parameters, *(ii)* calculate the loss function on
# the parameters (and concatenate), and last *(iii)* generate a new set of parameters using
# the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)

    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ensemble_kalman_process)

anim_unique_minimum = @animate for i in 1:N_iterations
    u_i = get_u(ensemble_kalman_process, i)

    plot(
        [u★[1]],
        [u★[2]],
        seriestype = :scatter,
        markershape = :star5,
        markersize = 11,
        markercolor = :red,
        label = "optimum u⋆",
    )

    plot!(
        u_i[1, :],
        u_i[2, :],
        seriestype = :scatter,
        xlims = extrema(u_init[1, :]),
        ylims = extrema(u_init[2, :]),
        xlabel = "u₁",
        ylabel = "u₂",
        markersize = 5,
        markeralpha = 0.6,
        markercolor = :blue,
        label = "particles",
        title = "EKI iteration = " * string(i),
    )
end
nothing # hide

# The results show that the minimizer of ``G_1`` is ``u=u_*``. 

gif(anim_unique_minimum, "unique_minimum_sparse.gif", fps = 1) # hide
