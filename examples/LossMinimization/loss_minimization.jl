# # Minimization Loss
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
# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
Random.seed!(rng_seed)
nothing # hide

# We choose the number of synthetic observations `n_obs` from ``G(u)``,
n_obs = 1

# and also set the observation noise level,
noise_level =  1e-8   

# Independent noise for synthetic observations       
Γy = noise_level * Matrix(I, n_obs, n_obs) 
noise = MvNormal(zeros(n_obs), Γy)

# We start here with an example using a loss function `G₁(u)` that has one minimum. Let's
# take the minimum to be at ``u_* = (-1, 1)``:
u★ = [1, -1]
nothing # hide

# and the loss function ``G₁(u) = \|u - u_*\|``, 
G₁(u) = [sqrt((u[1] - u★[1])^2 + (u[2] - u★[2])^2)]

y_obs  = G₁(u★)

# Define Prior
prior_distns = [Parameterized(Normal(0., sqrt(1))),
                Parameterized(Normal(-0., sqrt(1)))]
                
constraints = [[no_constraint()], [no_constraint()]]

prior_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distns, constraints, prior_names)

prior_mean = get_mean(prior)
prior_cov  = get_cov(prior)

# ### Calibration
#
# We choose the number of ensemble members,
N_ens = 50

# and the number of EKI iterations
N_iter = 20

# With that in hand, we can construct our initial ensemble
initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

# and the EKI
ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())

# Then we calibrate:
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    
    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ens]...)
    
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

# Now let's do an example in which the loss function has two minima. The procedure is the\
# same as before.

# Again, we set the seed for pseudo-random number generator for reproducibility,
rng_seed = 10 # 10 converges to one minima 100 converges to the other
Random.seed!(rng_seed)
nothing # hide

# The two loss function minima are at: ``u_{1*} = (1, -1)`` and ``u_{2*} = (-1, -1)``:
u₁★ = [ 1, -1]
u₂★ = [-1, -1]
nothing # hide

# and the loss function ``G₂(u)``:
G₂(u) = [abs((u[1] - u₁★[1]) * (u[1] - u₂★[1]))^2 + (u[2] - u₁★[2])^2]
G₂(u₁★)[1] == G₂(u₂★)[1]

y_obs  = [0.0]

# Define Prior
prior_distns = [Parameterized(Normal(0., sqrt(2))),
                Parameterized(Normal(-0., sqrt(2)))]
                
constraints = [[no_constraint()], [no_constraint()]]

prior_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distns, constraints, prior_names)

prior_mean = get_mean(prior)

prior_cov = get_cov(prior)

# ### Calibration
# We choose the number of ensemble members, the number of EKI iterations, construct our
# initial ensemble and the EKI:
N_ens = 50

N_iter = 40

initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())

# Then we calibrate:
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
