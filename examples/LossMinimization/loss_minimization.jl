# # Minimization of simple loss functions
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
# where ``u`` is a 2-vector of parameters and ``u_*`` is given; here ``u_* = (-1, 1)``.
u★ = [1, -1]
G₁(u) = [sqrt((u[1] - u★[1])^2 + (u[2] - u★[2])^2)]
nothing # hide


# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
Random.seed!(rng_seed)
nothing # hide
# We set a noise level, here this is just used to stabilize the algorithm
dim_output = 1
stabilization_level = 1e-3
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output) 

# We get the data by observing the function value at the true parameters
y_obs  = G₁(u★)

# ### Prior distributions
#
# As we work with a Bayesian method, we define the prior. This will behave like an "initial guess" for the likely region of parameter space we expect the solution to live in.
prior_distributions = [Parameterized(Normal(0, 1)), Parameterized(Normal(0, 1))]
                
constraints = [[no_constraint()], [no_constraint()]]

parameter_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distributions, constraints, parameter_names)

# ### Calibration
#
# We choose the number of ensemble members and the number of iterations of the algorithm
N_ensemble   = 20
N_iterations = 10
nothing # hide

# The initial ensemble is constructed by sampling the prior
initial_ensemble =
    EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ensemble;
                                                           rng_seed=rng_seed)

# We then initialize the Ensemble Kalman Process algorithm, with the initial ensemble, the data, the stabilization and the process: for EKI this is `Inversion`. EKI is just one choice of the available methods, and we initialize it by constructing the `Inversion` struct with `Inversion()`. and the EKI. The EKI is a choice of the available methods and it is constructed by

ensemble_kalman_process = 
    EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble, y_obs,
                                                      Γ_stabilization, Inversion())

# Then we calibrate by *(i)* obtaining the parameters, *(ii)* calculate the loss function on
# the parameters (and concatenate), and last *(iii)* generate a new set of parameters using
# the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)
    
    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ensemble]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ensemble_kalman_process, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ensemble_kalman_process)

anim_unique_minimum = @animate for i in 1:N_iterations
    u_i = get_u(ensemble_kalman_process, i)
    
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
# G₂(u) = \|u - v_{*}\| \|u - w_{*}\| ,
# ```
# where again ``u`` is a vector of parameters and ``v_{1*}`` and ``w_{*}`` are vectors of
# optimal parameters. Here, we take ``v_{*} = (1, -1)`` and ``w_{*} = (-1, -1)``.

v★ = [ 1, -1]
w★ = [-1, -1]
G₂(u) = [sqrt(((u[1] - v★[1])^2 + (u[2] - v★[2])^2)*((u[1] - w★[1])^2 + (u[2] - w★[2])^2))]
nothing # hide
#
# The procedure is same as the single-minimum example above.

# Again, we set the seed for pseudo-random number generator for reproducibility,
rng_seed = 10
Random.seed!(rng_seed)
nothing # hide

# We get the data by observing the function value at the true parameters
y_obs = G₂(w★)

# We choose the stabilization as in the single-mimum example

# ### Prior distributions
#
# We define the prior. We demonstrate here how we can place prior information through a bias of e.g., ``u₁``, showing we believe that ``u₁`` is more likely to be negatively valued. This can be implemented by setting the mean of the prior of its distribution `-0.5`:

prior_distributions = [Parameterized(Normal(-0.5, sqrt(2))),
                       Parameterized(Normal(   0, sqrt(2)))]
                
constraints = [[no_constraint()], [no_constraint()]]

parameter_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distributions, constraints, parameter_names)

# ### Calibration
#
# We choose the number of ensemble members, the number of EKI iterations, construct our initial ensemble and the EKI with the `Inversion()` constructor (exactly as in the single-minimum example):
N_ensemble   = 20
N_iterations = 20

initial_ensemble =
    EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ensemble;
                                                           rng_seed=rng_seed)

ensemble_kalman_process = 
    EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble, y_obs,
                                                      Γ_stabilization, Inversion())

# We calibrate again. Doing so involves *(i)* obtaining the parameters, *(ii)* calculating the
# loss function on the parameters (and concatenate), and last *(iii)* generate a new set of
# parameters using the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)
    
    g_ens = hcat([G₂(params_i[:, i]) for i in 1:N_ensemble]...)
    
    EnsembleKalmanProcessModule.update_ensemble!(ensemble_kalman_process, g_ens)
end

# and visualize the results:
u_init = get_u_prior(ensemble_kalman_process)

anim_two_minima = @animate for i in 1:N_iterations
    u_i = get_u(ensemble_kalman_process, i)

    plot([v★[1]], [v★[2]],
            seriestype = :scatter,
           markershape = :star5,
            markersize = 11,
           markercolor = :red,
                 label = "optimum v⋆")
     
    plot!([w★[1]], [w★[2]],
            seriestype = :scatter,
           markershape = :star5,
            markersize = 11,
           markercolor = :green,
                 label = "optimum w⋆")

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
