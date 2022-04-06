# # Minimization of simple loss functions
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
# where ``u`` is a 2-vector of parameters and ``u_*`` is given; here ``u_* = (-1, 1)``. 
u★ = [1, -1]
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
prior_u1 = Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "u1")
prior_u2 = Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => no_constraint(), "name" => "u2")

prior = ParameterDistribution([prior_u1, prior_u2])

# ### Calibration
#
# We choose the number of ensemble members and the number of iterations of the algorithm
N_ensemble = 20
N_iterations = 10
nothing # hide

# The initial ensemble is constructed by sampling the prior
initial_ensemble = EKP.construct_initial_ensemble(prior, N_ensemble; rng_seed = rng_seed)

# We then initialize the Ensemble Kalman Process algorithm, with the initial ensemble, the
# target, the stabilization and the process type (for EKI this is `Inversion`, initialized 
# with `Inversion()`). 
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, G_target, Γ_stabilization, Inversion())

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

gif(anim_unique_minimum, "unique_minimum.gif", fps = 1) # hide

# ## Loss function with two minima

# Now let's do an example in which the loss function has two minima. We minimize the loss
# function
# ```math
# G₂(u) = \|u - v_{*}\| \|u - w_{*}\| ,
# ```
# where again ``u`` is a 2-vector, and ``v_{*}`` and ``w_{*}`` are given 2-vectors. Here, we take ``v_{*} = (1, -1)`` and ``w_{*} = (-1, -1)``.

v★ = [1, -1]
w★ = [-1, -1]
G₂(u) = [sqrt(((u[1] - v★[1])^2 + (u[2] - v★[2])^2) * ((u[1] - w★[1])^2 + (u[2] - w★[2])^2))]
nothing # hide
#
# The procedure is same as the single-minimum example above.

# We set the seed for pseudo-random number generator for reproducibility,
rng_seed = 10
Random.seed!(rng_seed)
nothing # hide

# A positive function can be minimized with a target of 0,
G_target = [0]

# We choose the stabilization as in the single-minimum example

# ### Prior distributions
#
# We define the prior. We can place prior information on e.g., ``u₁``, demonstrating a belief
# that ``u₁`` is more likely to be negative. This can be implemented by setting a bias in the
# mean of its prior distribution to e.g., ``-0.5``:
prior_u1 = Dict("distribution" => Parameterized(Normal(-0.5, sqrt(2))), "constraint" => no_constraint(), "name" => "u1")
prior_u2 = Dict("distribution" => Parameterized(Normal(0, sqrt(2))), "constraint" => no_constraint(), "name" => "u2")

prior = ParameterDistribution([prior_u1, prior_u2])

# ### Calibration
#
# We choose the number of ensemble members, the number of EKI iterations, construct our initial ensemble and the EKI with the `Inversion()` constructor (exactly as in the single-minimum example):
N_ensemble = 20
N_iterations = 20

initial_ensemble = EKP.construct_initial_ensemble(prior, N_ensemble; rng_seed = rng_seed)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, G_target, Γ_stabilization, Inversion())

# We calibrate by *(i)* obtaining the parameters, *(ii)* calculating the
# loss function on the parameters (and concatenate), and last *(iii)* generate a new set of
# parameters using the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)

    g_ens = hcat([G₂(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ensemble_kalman_process)

anim_two_minima = @animate for i in 1:N_iterations
    u_i = get_u(ensemble_kalman_process, i)

    plot(
        [v★[1]],
        [v★[2]],
        seriestype = :scatter,
        markershape = :star5,
        markersize = 11,
        markercolor = :red,
        label = "optimum v⋆",
    )

    plot!(
        [w★[1]],
        [w★[2]],
        seriestype = :scatter,
        markershape = :star5,
        markersize = 11,
        markercolor = :green,
        label = "optimum w⋆",
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

# Our bias in the prior shifts the initial ensemble into the negative ``u_1`` direction, and
# thus increases the likelihood (over different instances of the random number generator) of
# finding the minimizer ``u=w_*``.

gif(anim_two_minima, "two_minima.gif", fps = 1) # hide
