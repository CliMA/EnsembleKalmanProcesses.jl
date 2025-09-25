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

# ## Loss function with single minimum
#
# Here, we minimize the loss function
# ```math
# G₁(u) = \|u - u_*\| ,
# ```
# where ``u`` is a 2-vector of parameters and ``u_*`` is given; here ``u_* = (-1, 1)``. 

# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)
nothing # hide

# We set a stabilization level, which can aid the algorithm convergence
dim_output = 1
stabilization_level = 1e-1
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output)
inv_Γ_stabilization = inv(Γ_stabilization)

ustar = [1, -1]
function G(u; stoch = false)
    if stoch
        return [norm(u - ustar)] + sqrt(stabilization_level) * randn(1)
    else
        return [norm(u - ustar)]
    end
end

nothing # hide

# The functional is positive so to minimize it we may set the target to be 0,
G_target = [0.0]
nothing # hide

# In this example we run several cases. The EKI ("inversion"), ETKI ("transform") find the mode of the posterior, while the EKS ("sampler") samples an approximation of the Gaussian spread of the posterior too. The EKI/ETKI have two variants, a variant which finds the mode of the posterior at algorithm time 1 ("finite"), and a variante which finds the mode of the posterior at algorithm time ∞ ("infinite"). The latter has an additional flexibility in that the initial ensemble does not need to be sampled at the prior.
cases = [
    "inversion-finite",
    "inversion-infinite",
    "transform-finite",
    "transform-infinite",
    "unscented-infinite",
    "transform-unscented-infinite",
    "sampler", # "aldi" variant of EKS
    "sampler-eks", # original EKS
    "gauss-newton",
]
case_list = cases[1:9] # i:j even if i=j

# We can choose to add noise to every "G" call? (making the loss function of the problem noisy)
stoch_G_flag = false

# and whether we produce animations
anim_flag = true

@info "add stochastic noise to G evaluations?: $(stoch_G_flag)"
# ### Prior distributions
#
# As we work with a Bayesian method, we define a prior. This will behave like an "initial guess"
# for the likely region of parameter space we expect the solution to live in. Here we define
# ``Normal(3,1)`` distributions with no constraints 
prior_u1 = constrained_gaussian("u1", 3, 1, -Inf, Inf)
prior_u2 = constrained_gaussian("u2", 3, 1, -Inf, Inf)
prior = combine_distributions([prior_u1, prior_u2])
nothing # hide
# !!! note

#  In this example there are no constraints, therefore no parameter transformations.
anim_skip = 1


# The initial ensemble is constructed by sampling 20 particles. The "finite" cases must sample these from the prior, while the "infinite" cases do not. We choose an off-centered distribution to illustrate this property (centered at (-2,4)).

N_ensemble = 20

u_trajs = []
for case in case_list
    @info "Running case $case"
    if case ∈ ["inversion-finite", "transform-finite"]
        initial_ensemble = EKP.construct_initial_ensemble(copy(rng), prior, N_ensemble)
    elseif case ∈ ["unscented-finite", "transform-unscented-finite"]
        initial_u1 = constrained_gaussian("u1", -2, 1, -Inf, Inf)
        initial_u2 = constrained_gaussian("u2", 4, sqrt(2.0), -Inf, Inf)
        initial_dist = combine_distributions([initial_u1, initial_u2])
    else # doesn't need to sample the prior
        initial_u1 = constrained_gaussian("u1", -2, 1, -Inf, Inf)
        initial_u2 = constrained_gaussian("u2", 4, sqrt(2.0), -Inf, Inf)
        initial_dist = combine_distributions([initial_u1, initial_u2])
        initial_ensemble = EKP.construct_initial_ensemble(copy(rng), initial_dist, N_ensemble)
    end

    if case == "inversion-finite"
        process = Inversion()
        scheduler = DataMisfitController(terminate_at = 1) # =1
        N_iterations = 200
    elseif case == "inversion-infinite"
        process = Inversion(prior) # given the prior to impose
        scheduler = DataMisfitController(terminate_at = 100)# >>1
        N_iterations = 200
    elseif case == "transform-finite"
        process = TransformInversion()
        scheduler = DataMisfitController(terminate_at = 1) # =1
        N_iterations = 200
    elseif case == "transform-infinite"
        process = TransformInversion(prior)
        scheduler = DataMisfitController(terminate_at = 100) # =1
        N_iterations = 200
    elseif case == "unscented-infinite"
        process = Unscented(prior; impose_prior = true)
        scheduler = DataMisfitController(terminate_at = 100) # =1
        N_iterations = 200
    elseif case == "transform-unscented-infinite"
        process = TransformUnscented(prior; impose_prior = true)
        scheduler = DataMisfitController(terminate_at = 100) # =1
        N_iterations = 200
        #        N_iterations = 5
    elseif case == "sampler"
        process = Sampler(prior)
        #fixed_step = 1e-3 # 2e-6 unstable
        scheduler = EKSStableScheduler()
        N_iterations = 200
    elseif case == "sampler-eks"
        process = Sampler(prior, sampler_type = "eks")
        #fixed_step = 1e-3 # 2e-6 unstable
        scheduler = EKSStableScheduler()
        N_iterations = 200
    elseif case == "gauss-newton"
        process = GaussNewtonInversion(prior)
        scheduler = DataMisfitController(terminate_at = 100) # =1
        N_iterations = 200
    else
        println(case)
        throw(ArgumentError("Case not implemented yet"))
    end

    # We then initialize the Ensemble Kalman Process algorithm, with the initial ensemble, the
    # target, the stabilization and the process type (for EKI this is `Inversion`, initialized 
    # with `Inversion()`). We also remove the cutting-edge defaults and instead use the vanilla options.
    if case ∈ ["unscented-infinite", "transform-unscented-infinite"]
        ensemble_kalman_process = EKP.EnsembleKalmanProcess(
            G_target,
            Γ_stabilization,
            process,
            scheduler = scheduler,
            accelerator = DefaultAccelerator(),
            localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
            verbose = true,
        )
    else
        ensemble_kalman_process = EKP.EnsembleKalmanProcess(
            initial_ensemble,
            G_target,
            Γ_stabilization,
            process,
            scheduler = scheduler,
            accelerator = DefaultAccelerator(),
            localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
            verbose = true,
        )
    end
    nothing # hide
    # Then we calibrate by *(i)* obtaining the parameters, *(ii)* calculate the loss function on
    # the parameters (and concatenate), and last *(iii)* generate a new set of parameters using
    # the model outputs:
    N_iter = [N_iterations]
    for i in 1:N_iterations
        params_i = get_u_final(ensemble_kalman_process)
        N_ens = get_N_ens(ensemble_kalman_process)
        g_ens = hcat([G(params_i[:, i], stoch = stoch_G_flag) for i in 1:N_ens]...)

        terminate = EKP.update_ensemble!(ensemble_kalman_process, g_ens)
        if !isnothing(terminate)
            N_iter[1] = i
            break
        end

    end
    # and visualize the results:

    # plot posterior exact (2D)
    plotrange = collect(-2:0.05:5)
    #    plot_square = [[x,y] for x in plotrange, y in plotrange]

    function post_potential(uu, ff, yy, inv_Γ, mm, inv_C)
        return exp(-0.5 * (yy - ff)' * inv_Γ * (yy - ff) - 0.5 * (uu - mm)' * inv_C * (uu - mm))
    end
    function prior_potential(uu, mm, inv_C)
        return exp(-0.5 * (uu - mm)' * inv_C * (uu - mm))
    end
    PP_unnorm = zeros(length(plotrange), length(plotrange))
    VV_unnorm = zeros(length(plotrange), length(plotrange))
    prior_mean = mean(prior)
    prior_cov = cov(prior)
    inv_prior_cov = inv(prior_cov)
    for (i, pr1) in enumerate(plotrange)
        for (j, pr2) in enumerate(plotrange)
            VV_unnorm[i, j] = post_potential(
                [pr1, pr2],
                G([pr1, pr2], stoch = stoch_G_flag),
                G_target,
                inv_Γ_stabilization,
                prior_mean,
                inv_prior_cov,
            )
            PP_unnorm[i, j] = prior_potential([pr1, pr2], prior_mean, inv_prior_cov)
        end
    end

    # normalization
    using Trapz
    ZZ = trapz((plotrange, plotrange), VV_unnorm)
    VV = VV_unnorm / ZZ
    pZZ = trapz((plotrange, plotrange), PP_unnorm)
    PP = PP_unnorm / pZZ

    u_init = get_u_prior(ensemble_kalman_process)



    # and visualize the results:
    u_init = get_u_prior(ensemble_kalman_process)
    u_final = get_u_final(ensemble_kalman_process)

    # fixed image
    plt = plot(
        u_final[1, :],
        u_final[2, :],
        seriestype = :scatter,
        xlims = [minimum(plotrange), maximum(plotrange)],
        ylims = [minimum(plotrange), maximum(plotrange)],
        xlabel = "u₁",
        ylabel = "u₂",
        markersize = 5,
        markeralpha = 0.6,
        markercolor = :blue,
        label = "final",
        legend = true,
    )
    plot!(
        plt,
        [ustar[1]],
        [ustar[2]],
        seriestype = :scatter,
        markershape = :star5,
        markersize = 11,
        markercolor = :red,
        label = "optimum u⋆",
    )
    plot!(
        plt,
        u_init[1, :],
        u_init[2, :],
        seriestype = :scatter,
        markersize = 5,
        markeralpha = 0.6,
        markercolor = :red,
        label = "initial",
    )

    contour!(plt, plotrange, plotrange, VV', levels = exp.(collect(-5:1:5)), cbar = false)
    contour!(plt, plotrange, plotrange, PP', levels = exp.(collect(-5:1:5)), color = :darkred, cbar = false)

    figure_path = "final_iteration_$case.png"
    savefig(plt, figure_path)
    @info "saved figure in $(joinpath(@__DIR__,figure_path))"
    if anim_flag
        anim = @animate for i in 1:anim_skip:N_iter[1]
            u_i = get_u(ensemble_kalman_process, i)

            plot(
                [ustar[1]],
                [ustar[2]],
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
                xlims = [minimum(plotrange), maximum(plotrange)],
                ylims = [minimum(plotrange), maximum(plotrange)],
                xlabel = "u₁",
                ylabel = "u₂",
                markersize = 5,
                markeralpha = 0.6,
                markercolor = :blue,
                label = "particles",
                title = "iteration = " * string(i),
            )

            contour!(plotrange, plotrange, VV', levels = exp.(collect(-5:1:5)), cbar = false, color = :blue)
            contour!(plotrange, plotrange, PP', levels = exp.(collect(-5:1:5)), color = :red, cbar = false)


        end
        nothing # hide

        # The results show that the minimizer of ``G_1`` is ``u=u_*``. 

        if stoch_G_flag
            gif(anim, "animated_$(case)_stochG.gif", fps = 10) # hide
        else
            gif(anim, "animated_$case.gif", fps = 10) # hide
        end
    end

    push!(u_trajs, get_u(ensemble_kalman_process))

end


# plot trajectories
u_means = []
u_stds = []
for u_traj in u_trajs
    push!(u_means, [mean(uu, dims = 2) for uu in u_traj])
    push!(u_stds, [std(uu, dims = 2) for uu in u_traj])
end
pp = plot(legend = true)
for (i, case) in enumerate(case_list)
    colors = palette(:tab10)
    u_mean_diff = [norm(u_means[i][idx] - ustar) for idx in 1:length(u_means[i])]
    u_stds_size = [norm(u_stds[i][idx]) for idx in 1:length(u_stds[i])]
    plot!(pp, 1:length(u_means[i]), u_mean_diff, label = case, color = colors[i], lw = 3)
    plot!(pp, 1:length(u_stds[i]), u_stds_size, label = case, linestyle = :dash, color = colors[i])
end

if stoch_G_flag
    savefig(pp, "mean_over_iteration_stochG.png")
else
    savefig(pp, "mean_over_iteration.png")
end
