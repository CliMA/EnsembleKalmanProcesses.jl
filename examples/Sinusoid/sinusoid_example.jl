# # [Fitting parameters of a sinusoid](@id sinusoid-example)
#
# !!! info "How do I run this code?"
#    The full code is found in the [`examples/`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples) directory of the github repository
#
# In this example we have a model that produces a sinusoid
# ``f(A, v) = A \sin(\phi + t) + v, \forall t \in [0,2\pi]``, with a random
# phase ``\phi``. Given an initial guess of the parameters as
# ``A^* \sim \mathcal{N}(2,1)`` and ``v^* \sim \mathcal{N}(0,25)``, our goal is
# to estimate the parameters from a noisy observation of the maximum, minimum,
# and mean of the true model output.

# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
nothing # hide

## Setting up the model and data for our inverse problem

# Now, we define a model which generates a sinusoid given parameters ``\theta``: an
# amplitude and a vertical shift. We will estimate these parameters from data.
# The model adds a random phase shift upon evaluation.
dt = 0.01
trange = 0:dt:(2 * pi + dt)
function model(amplitude, vert_shift)
    phi = 2 * pi * rand(rng)
    return amplitude * sin.(trange .+ phi) .+ vert_shift
end
nothing # hide

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
nothing # hide

# We then define ``G(\theta)``, which returns the observables of the sinusoid
# given a parameter vector. These observables should be defined such that they
# are informative about the parameters we wish to estimate. Here, the two
# observables are the ``y`` range of the curve (which is informative about its
# amplitude), as well as its mean (which is informative about its vertical shift).
function G(u)
    theta, vert_shift = u
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end
nothing # hide

# Suppose we have a noisy observation of the true system. Here, we create a
# pseudo-observation ``y`` by running our model with the correct parameters
# and adding Gaussian noise to the output.
dim_output = 2

Γ = 0.1 * I
noise_dist = MvNormal(zeros(dim_output), Γ)

theta_true = [1.0, 7.0]
y = G(theta_true) .+ rand(noise_dist)
nothing # hide

## Solving the inverse problem

# We now define prior distributions on the two parameters. For the amplitude,
# we define a prior with mean 2 and standard deviation 1. It is
# additionally constrained to be nonnegative. For the vertical shift we define
# a Gaussian prior with mean 0 and standard deviation 5.
prior_u1 = constrained_gaussian("amplitude", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("vert_shift", 0, 5, -Inf, Inf)
prior = combine_distributions([prior_u1, prior_u2])
nothing # hide

# We now generate the initial ensemble and set up the ensemble Kalman inversion.
N_ensemble = 5
N_iterations = 5

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)
nothing # hide

# We are now ready to carry out the inversion. At each iteration, we get the
# ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
# and apply the Kalman update to the ensemble.
for i in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end
nothing # hide

# Finally, we get the ensemble after the last iteration. This provides our estimate of the parameters.
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# To visualize the success of the inversion, we plot model with the true
# parameters, the initial ensemble, and the final ensemble.
p = plot(trange, model(theta_true...), c = :black, label = "Truth", legend = :bottomright, linewidth = 2)
plot!(
    p,
    trange,
    [model(get_ϕ(prior, ensemble_kalman_process, 1)[:, i]...) for i in 1:N_ensemble],
    c = :red,
    label = ["Initial ensemble" "" "" "" ""],
)
plot!(
    p,
    trange,
    [model(final_ensemble[:, i]...) for i in 1:N_ensemble],
    c = :blue,
    label = ["Final ensemble" "" "" "" ""],
)

xlabel!("Time")

# We see that the final ensemble is much closer to the truth. Note that the
# random phase shift is of no consequence.

savefig(p, "output.png")
