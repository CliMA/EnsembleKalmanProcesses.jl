# # [Fitting parameters of a sinusoid](@id sinusoid-example)

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

# Seed for pseudo-random number generator.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
nothing # hide

# make gif (slow)
make_gif = false

# Choose a case
cases = ["inversion", "sampler", "nonrev_sampler"]
case = cases[2]

if case == "inversion"
    process = Inversion()
elseif case == "sampler"
    process = Sampler(prior)
elseif case == "nonrev_sampler"
    process = NonreversibleSampler(prior, prefactor = 1.5) # prefactor (1.1 - 1.5) vs stepsize
end
# some methods have better than fixed timesteppers
fixed_step = 1e-3
scheduler = DefaultScheduler(fixed_step)
# scheduler = EKSStableScheduler()   
# scheduler = DataMisfitController()


# We then define ``G(\theta)``, which returns the observables of the sinusoid
# given a parameter vector. These observables should be defined such that they
# are informative about the parameters we wish to estimate. Here, the two
# observables are the ``y`` range of the curve (which is informative about its
# amplitude), as well as its mean (which is informative about its vertical shift).
function G(u)
    return u[1] + u[2]
end
nothing # hide

# Suppose we have a noisy observation of the true system. Here, we create a
# pseudo-observation ``y`` by running our model with the correct parameters
# and adding Gaussian noise to the output.
dim_output = 2

Γ = 1.0 * I
noise_dist = MvNormal(zeros(dim_output), Γ)

theta_true = [1.0, 7.0]
y = [1.0] # G(theta_true) .+ rand(noise_dist)
nothing # hide

## Solving the inverse problem

# We now define prior distributions on the two parameters. For the amplitude,
# we define a prior with mean 2 and standard deviation 1. It is
# additionally constrained to be nonnegative. For the vertical shift we define
# a Gaussian prior with mean 0 and standard deviation 5.
prior = constrained_gaussian("u", 0, sqrt(10), -Inf, Inf, repeats = 2)
nothing # hide

# We now generate the initial ensemble and set up the ensemble Kalman inversion.
N_ensemble = 50
N_iterations = 2000

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(
    initial_ensemble,
    y,
    Γ,
    process;
    rng = rng,
    scheduler = DefaultScheduler(fixed_step),
    #   scheduler = DataMisfitController(),
    #     scheduler = EKSStableScheduler(),
    verbose = true,
)
nothing # hide

# We are now ready to carry out the inversion. At each iteration, we get the
# ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
# and apply the Kalman update to the ensemble.
for i in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    println(mean(params_i, dims = 2)[:])
    println(cov(params_i, dims = 2))
    G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

nothing # hide

# Finally, we get the ensemble after the last iteration. This provides our estimate of the parameters.
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)
# To visualize the success of the inversion, we plot model with the true
# parameters, the initial ensemble, and the final ensemble.
if make_gif
    anim_linear = @animate for i in 1:N_iterations

        ppp = scatter(
            get_ϕ(prior, ensemble_kalman_process, 1)[1, :],
            get_ϕ(prior, ensemble_kalman_process, 1)[2, :],
            c = :red,
            label = ["Initial ensemble" "" "" "" ""],
        )
        soln = get_ϕ(prior, ensemble_kalman_process, i)

        scatter!(ppp, soln[1, :], soln[2, :], c = :blue, label = ["ensemble $i"])
    end
    gif(anim_linear, "linear_output_$case.gif", fps = 30) # hide
end

ppp = scatter(
    get_ϕ(prior, ensemble_kalman_process, 1)[1, :],
    get_ϕ(prior, ensemble_kalman_process, 1)[2, :],
    c = :red,
    label = ["Initial ensemble" "" "" "" ""],
)

scatter!(
    ppp,
    get_ϕ_final(prior, ensemble_kalman_process)[1, :],
    get_ϕ_final(prior, ensemble_kalman_process)[2, :],
    c = :blue,
    label = ["Final ensemble" "" "" "" ""],
)
savefig(ppp, "linear_output_$case.png")


@info "final $(final_ensemble)"
@info "final mean $(mean(final_ensemble,dims=2)[:])"
@info "final cov $(cov(final_ensemble,dims=2))"
@info "final corr(u1,u2) $(cor(final_ensemble[1,:],final_ensemble[2,:]))"
