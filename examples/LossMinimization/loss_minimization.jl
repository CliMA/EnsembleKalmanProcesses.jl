# # Minimization Loss
#
# First we load the required packages.

using Distributions
using LinearAlgebra
using Random
using Plots
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

# Seed for pseudo-random number generator for reproducibility
rng_seed = 41
Random.seed!(rng_seed)
nothing # hide

# Number of synthetic observations from ``G(u)``
n_obs = 1

# Defining the observation noise level
noise_level =  1e-8   

# Independent noise for synthetic observations       
Γy = noise_level * Matrix(I, n_obs, n_obs) 
noise = MvNormal(zeros(n_obs), Γy)

# Loss Function minimum, ``u_* = (-1, 1)``:
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

prior_cov = get_cov(prior)

# Calibrate
N_ens = 50  # number of ensemble members
N_iter = 20 # number of EKI iterations
initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())
#
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    
    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ens]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)
end

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

# Now let's do a case in which the loss function has two minima.

# Again, we seed for pseudo-random number generator for reproducibility
rng_seed = 10 # 10 converges to one minima 100 converges to the other
Random.seed!(rng_seed)
nothing # hide

# Loss function minima: ``u_{1*} = (1, -1)`` and ``u_{2*} = (-1, -1)``:
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

# Calibrate
N_ens = 50  # number of ensemble members
N_iter = 40 # number of EKI iterations
initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())
#
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    
    g_ens = hcat([G₂(params_i[:, i]) for i in 1:N_ens]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)
end

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
