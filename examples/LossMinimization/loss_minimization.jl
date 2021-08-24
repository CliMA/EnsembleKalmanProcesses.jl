# # Minimization of Loss function
#
# First we load the required packages.

using
    Distributions,
    LinearAlgebra,
    Random,
    Plots

using
    EnsembleKalmanProcesses.EnsembleKalmanProcessModule,
    EnsembleKalmanProcesses.ParameterDistributionStorage

# ## Loss function with single minimum
#
# Here, we minimize the loss function
# ```math
# G₁(u) = \|u - u_*\| ,
# ```
# where ``u`` is a vector of parameters and ``u_*`` is a vector of optimal parameters, here
# taken to be ``u_* = (-1, 1)``.
#
# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
Random.seed!(rng_seed)
nothing # hide

# We choose the number of observations and noise level,
N_observations = 1
noise_level = 1e-3
nothing # hide

# Independent noise for synthetic observation:
Γ_stabilisation = noise_level * Matrix(I, N_observations, N_observations) 
noise = MvNormal(zeros(N_observations), Γ_stabilisation)

# The optimum parameters ``u_* = (-1, 1)``:
u★ = [1, -1]
nothing # hide

# and the loss function ``G₁``: 
G₁(u) = [sqrt((u[1] - u★[1])^2 + (u[2] - u★[2])^2)]

y_obs  = G₁(u★)

# ### Prior distributions
#
# We then define the prior
prior_distns = [Parameterized(Normal(0, 1)),
                Parameterized(Normal(0, 1))]
                
constraints = [[no_constraint()], [no_constraint()]]

prior_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distns, constraints, prior_names)

# ### Calibration
#
# We choose the number of ensemble members and the number of EKI iterations¨
N_ensemble  = 50
N_iterations = 20
nothing # hide

# With that in hand, we can construct our initial ensemble
initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

# and the EKI. The EKI is a choice of the available methods and it is constructed by
# initializing with `Inversion()` method.

ensemble_kalman_process =
    EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble, y_obs, Γ_stabilisation, Inversion())

# Then we calibrate by *(i)* obtaining the parameters, *(ii)* calculate the loss function on
# the parameters (and concatenate), and last *(iii)* generate a new set of parameters using
# the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)
    
    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ensemble]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ekiobj)

anim_unique_minimum = @animate for i in 1:N_iter
    u_i = get_u(ekiobj, i)
    
    plot([u★[1]], [u★[2]],
          seriestype = :scatter,
         markershape = :star5,
          markersize = 11,
         markercolor = :red,
               label = "optimum u⋆"
        )

    plot!(u_i[1, :], u_i[2, :],
             seriestype = :scatter,
                  xlims = extrema(u_init[1, :]),
                  ylims = extrema(u_init[2, :]),
                 xlabel = "u₁",
                 ylabel = "u₂",
             markersize = 5,
            markeralpha = 0.6,
            markercolor = :blue,
                  label = "particles",
                  title = "EKI iteration = " * string(i)
         )    
end

gif(anim_unique_minimum, "unique_minimum.gif", fps = 1) # hide

# ## Loss function with two minima

# Now let's do an example in which the loss function has two minima. We minimize the loss
# function
# ```math
# G₂(u) = \|u - u_{1*}\| \|u - u_{2*}\| ,
# ```
# where again ``u`` is a vector of parameters and ``u_{1*}`` and ``u_{2*}`` are vectors of
# optimal parameters. Here, we take ``u_{1*} = (1, -1)`` and ``u_{2*} = (-1, -1)``.
#
# The procedure is same as the single-minimum example above.

# Again, we set the seed for pseudo-random number generator for reproducibility,
rng_seed = 10
Random.seed!(rng_seed)
nothing # hide

# The two optimal set of parameter values are:
u₁★ = [ 1, -1]
u₂★ = [-1, -1]
nothing # hide

# and the loss function ``G₂``:
G₂(u) = [sqrt(((u[1] - u₁★[1])^2 + (u[2] - u₁★[2])^2)*((u[1] - u₂★[1])^2 + (u[2] - u₂★[2])^2))]

y_obs = [0.0]

# ### Prior distributions
#
# Now define the prior. Suppose we have a bias on the prior of ``u₁``, e.g., that we know that
# ``u₁`` is more likely to be negative. We take the mean of the prior of the first distribution
# as `Normal(-0.5, sqrt(2))`:

prior_distns = [Parameterized(Normal(-0.5, sqrt(2))),
                Parameterized(Normal(   0, sqrt(2)))]
                
constraints = [[no_constraint()], [no_constraint()]]

prior_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distns, constraints, prior_names)

# ### Calibration
#
# We choose the number of ensemble members, the number of EKI iterations, construct our
# initial ensemble and the EKI (similarly as in the single-minimum example):
N_ens  = 50
N_iter = 40

initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble, y_obs,
                                                           Γ_stabilisation, Inversion())

# We calibrate again. Doing so involves *(i)* obtaining the parameters, *(ii)* calculating the
# loss function on the parameters (and concatenate), and last *(iii)* generate a new set of
# parameters using the model outputs:
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    
    g_ens = hcat([G₂(params_i[:, i]) for i in 1:N_ens]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ekiobj)

anim_two_minima = @animate for i in 1:N_iter
    u_i = get_u(ekiobj, i)

    plot([u₁★[1]], [u₁★[2]],
            seriestype = :scatter,
           markershape = :star5,
            markersize = 11,
           markercolor = :red,
                 label = "optimum u₁⋆")
     
    plot!([u₂★[1]], [u₂★[2]],
            seriestype = :scatter,
           markershape = :star5,
            markersize = 11,
           markercolor = :green,
                 label = "optimum u₂⋆")

    plot!(u_i[1, :], u_i[2, :],
             seriestype = :scatter,
                  xlims = extrema(u_init[1, :]),
                  ylims = extrema(u_init[2, :]),
                 xlabel = "u₁",
                 ylabel = "u₂",
             markersize = 5,
            markeralpha = 0.6,
            markercolor = :blue,
                  label = "particles",
                  title = "EKI iteration = " * string(i)
         )
end

gif(anim_two_minima, "two_minima.gif", fps = 1) # hide
