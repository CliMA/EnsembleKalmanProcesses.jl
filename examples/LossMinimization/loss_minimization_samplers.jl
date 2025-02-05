# # Minimization of simple loss functions
#
# !!! info "How do I run this code?"
#     The full code is found in the [`examples/`](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples) directory of the github repository
#
# First we load the required packages.

using Distributions, LinearAlgebra, Random, Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

const EKP = EnsembleKalmanProcesses

cases = ["inversion", "sampler", "nonrev_sampler"]
case = cases[3]
antisymmetric_multipliers = [1]#[collect(0.1:0.1:1.0)...]

# user configurables:
N_ensemble = 20
anim_skip = 10


# ## Loss function with single minimum
#
# Here, we minimize the loss function
# ```math
# G₁(u) = \|u - u_*\| ,
# ```

# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)
nothing # hide

# We set a stabilization level, which can aid the algorithm convergence
dim_output = 1
stabilization_level = 1.0
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output)


# ### Prior distributions
#
# As we work with a Bayesian method, we define a prior. This will behave like an "initial guess"
# for the likely region of parameter space we expect the solution to live in. Here we define
# ``Normal(0,1)`` distributions with no constraints 
prior_u1 = constrained_gaussian("u1", -4, 1, -Inf, Inf)
prior_u2 = constrained_gaussian("u2", 4, 1, -Inf, Inf)
prior = combine_distributions([prior_u1, prior_u2])
nothing # hide
# !!! note
#     In this example there are no constraints, therefore no parameter transformations.

# ### Calibration
for am in antisymmetric_multipliers
    if case == "inversion"
        process = Inversion()
        fixed_step = 1e-6 # can use DMC for better approximation below
        #    scheduler = DefaultScheduler(fixed_step) 
        scheduler = DataMisfitController(terminate_at=10)# terminate in ~6 iterations
        N_iterations = 100# Int(ceil(10.0/fixed_step))

    elseif case == "sampler"
        process = Sampler(prior)
        fixed_step = 1e-4 # 2e-6 unstable
        scheduler = DefaultScheduler(fixed_step)
#        scheduler = EKSStableScheduler()
        N_iterations = 1000
        
    elseif case == "nonrev_sampler" # max dt = 5e-5
        process = NonreversibleSampler(prior, prefactor = 10, antisymmetric_multiplier = am) # prefactor (1.1 - 1.5) vs stepsize
    fixed_step = 1e-4 # 2e-6 unstable
    scheduler = DefaultScheduler(fixed_step)
#    scheduler = EKSStableScheduler(0.3,1e-6)
    N_iterations = 1000
        
    end
    
    # The initial ensemble is constructed by 
    #initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)
    initial_u1 = constrained_gaussian("u", 5, 1, -Inf, Inf)
    initial_u2 = constrained_gaussian("u2", 7, sqrt(2.0), -Inf, Inf)
    initial_dist = combine_distributions([initial_u1, initial_u2])
    initial_ensemble = EKP.construct_initial_ensemble(rng, initial_dist, N_ensemble)
    
    # ## Loss function with two minima
    
    # Now let's do an example in which the loss function has two minima. We minimize the loss
    # function
    # ```math
    # G₂(u) = \|u - v_{*}\| \|u - w_{*}\| ,
    # ```
    # where again ``u`` is a 2-vector, and ``v_{*}`` and ``w_{*}`` are given 2-vectors. Here, we take ``v_{*} = (1, -1)`` and ``w_{*} = (-1, -1)``.
    u★ = [1, -1]
    G₁(u) = [(u[1] - u★[1])^2 + (u[2] - u★[2])^2]
    G_target = [0]
    nothing # hide
    
    v★ = [4, -4]
    w★ = [-4, 4]
    G₂(u) = norm(u - v★) * norm(u - w★)
    nothing # hide

    v = [1, -1]
    Gnew(u) = (u[1] - v[1]) + (u[2] - v[2])^2
    
    #
    # The procedure is same as the single-minimum example above.
        
    # A positive function can be minimized with a target of 0,
    G_target = [0]
    
    # We choose the stabilization as in the single-minimum example

    # plot posterior
    
    
    
    # ### Calibration
    #
    # We choose the number of ensemble members, the number of EKI iterations, construct our initial ensemble and the EKI with the `Inversion()` constructor (exactly as in the single-minimum example):
    
    ensemble_kalman_process = EKP.EnsembleKalmanProcess(
        initial_ensemble,
        G_target,
        Γ_stabilization,
        process,
        scheduler = scheduler,
        accelerator = DefaultAccelerator(),
        localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
        verbose=true,
    )

    # We calibrate by *(i)* obtaining the parameters, *(ii)* calculating the
    # loss function on the parameters (and concatenate), and last *(iii)* generate a new set of
    # parameters using the model outputs:
    for i in 1:N_iterations
        params_i = get_ϕ_final(prior, ensemble_kalman_process)
        
        g_ens = hcat([G₂(params_i[:, i]) for i in 1:N_ensemble]...)
        @info mean(g_ens, dims=2)
        EKP.update_ensemble!(ensemble_kalman_process, g_ens)
    end
    
    # and visualize the results:
    u_init = get_u_prior(ensemble_kalman_process)
    
    anim_two_minima = @animate for i in 1:anim_skip:N_iterations
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
#            xlims = [min(v★[1],w★[1])-5, max(v★[1],w★[1])+5],
#            ylims = [min(v★[2],w★[2])-5, max(v★[2],w★[2])+5],
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
    
    if isa(process, NonreversibleSampler)
        gif(anim_two_minima, "two_minima_$(case)_multiplier_$(am).gif", fps = 30) # hide
    else
        gif(anim_two_minima, "two_minima_$(case).gif", fps = 30) # hide
    end
end
