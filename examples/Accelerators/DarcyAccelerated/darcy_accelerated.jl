# We will solve for an unknown permeability field ``\kappa`` governing the pressure of a Darcy flow on a square 2D domain. To learn about the permeability we shall take few pointwise measurements of the solved pressure field within the domain. 
# The forward solver is a simple finite difference scheme taken and modified from code [here](https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl). 

using LinearAlgebra
using Distributions
using Random
using JLD2
using LaTeXStrings

# the package to define the function distributions
import GaussianRandomFields # we wrap this so we don't want to use "using"
const GRF = GaussianRandomFields

# and finally the EKP packages
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

# We include the forward solver here
include("GModel.jl")

# Then link some outputs for figures and plotting
fig_save_directory = joinpath(@__DIR__, "output")
data_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end

PLOT_FLAG = true
if PLOT_FLAG
    using Plots
    @info "Plotting enabled, this will reduce code performance. Figures stored in $fig_save_directory"
end

function main()
    # Set a random seed.
    seed = 100234
    rng = Random.MersenneTwister(seed)

    cases = ["ens52-step1e-2_check", "ens200-step1e-2", "ens10-step1e-2"]
    case = cases[1]

    @info "running case $case"
    if case == "ens52-step1e-2_check"
        scheduler = DefaultScheduler(0.01)
        N_ens = 52
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "ens200-step1e-2"
        scheduler = DefaultScheduler(0.01)  #DataMisfitController(terminate_at = 1e4)
        N_ens = 200
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "ens10-step1e-2"
        scheduler = DefaultScheduler(0.01) #DataMisfitController(terminate_at = 1e2)
        N_ens = 10
        localization_method = EKP.Localizers.NoLocalization() #.SEC(1.0, 0.01)
    end

    # Define the spatial domain and discretization 
    dim = 2
    N, L = 80, 1.0
    pts_per_dim = LinRange(0, L, N)
    obs_ΔN = 10

    # To provide a simple test case, we assume that the true function parameter is a particular sample 
    # from the function space we set up to define our prior. More precisely we choose a value of the 
    # truth that doesnt have a vanishingly small probability under the prior defined by a probability 
    # distribution over functions; here taken as a family of Gaussian Random Fields (GRF). 
    # The function distribution is characterized by a covariance function - here a Matern kernel which 
    # assumes a level of smoothness over the samples from the distribution. We define an appropriate 
    # expansion of this distribution, here based on the Karhunen-Loeve expansion (similar to an 
    # eigenvalue-eigenfunction expansion) that is truncated to a finite number of terms, known as the 
    # degrees of freedom (`dofs`). The `dofs` define the effective dimension of the learning problem, 
    # decoupled from the spatial discretization. Explicitly, larger `dofs` may be required to represent 
    # multiscale functions, but come at an increased dimension of the parameter space and therefore a
    # typical increase in cost and difficulty of the learning problem.
    smoothness = 1.0
    corr_length = 0.25
    dofs = 50
    grf = GRF.GaussianRandomField(
        GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
        GRF.KarhunenLoeve(dofs),
        pts_per_dim,
        pts_per_dim,
    )

    # We define a wrapper around the GRF, and as the permeability field must be positive we introduce a domain constraint into the function distribution. Henceforth, the GRF is interfaced in the same manner as any other parameter distribution with regards to interface.
    pkg = GRFJL()
    distribution = GaussianRandomFieldInterface(grf, pkg) # our wrapper from EKP
    domain_constraint = bounded_below(0) # make κ positive
    pd = ParameterDistribution(
        Dict("distribution" => distribution, "name" => "kappa", "constraint" => domain_constraint),
    ) # the fully constrained parameter distribution

    # Now we have a function distribution, we sample a reasonably high-probability value from this distribution as a true value (here all degrees of freedom set with `u_{\mathrm{true}} = -0.5`). We use the EKP transform function to build the corresponding instance of the ``\kappa_{\mathrm{true}}``.
    u_true = -1.5 * ones(dofs, 1) # the truth parameter
    κ_true = transform_unconstrained_to_constrained(pd, u_true) # builds and constrains the function.  
    κ_true = reshape(κ_true, N, N)

    # Now we generate the data sample for the truth in a perfect model setting by evaluating the the model here, and observing it by subsampling in each dimension every `obs_ΔN` points, and add some observational noise
    darcy = Setup_Param(pts_per_dim, obs_ΔN, κ_true)
    h_2d = solve_Darcy_2D(darcy, κ_true)
    y_noiseless = compute_obs(darcy, h_2d)
    obs_noise_cov = 0.05^2 * I(length(y_noiseless)) * (maximum(y_noiseless) - minimum(y_noiseless))

    # Now we set up the Bayesian inversion algorithm. The prior we have already defined to construct our truth
    prior = pd

    # We define some algorithm parameters, here we take ensemble members larger than the dimension of the parameter space
    #N_ens = dofs + 2    # number of ensemble members
    N_iter = 100         # number of EKI iterations
    N_trials = 10       # number of trials 
    @info "obtaining statistics over $N_trials trials"

    errs = zeros(N_trials, N_iter)
    errs_acc = zeros(N_trials, N_iter)
    errs_acc_cs = zeros(N_trials, N_iter)

    for (idx, trial) in enumerate(1:N_trials)

        ytrial = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))
        observation = Observation(Dict("samples" => ytrial, "covariances" => obs_noise_cov, "names" => ["y"]))

        @info "computing trial $idx"
        # We sample the initial ensemble from the prior, and create three EKP objects to 
        # perform EKI algorithm using three different acceleration methods.
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_params,
            observation,
            Inversion(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
            accelerator = DefaultAccelerator(),
        )
        ekiobj_acc = EKP.EnsembleKalmanProcess(
            initial_params,
            observation,
            Inversion(),
            accelerator = NesterovAccelerator(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_acc_cs = EKP.EnsembleKalmanProcess(
            initial_params,
            observation,
            Inversion(),
            accelerator = FirstOrderNesterovAccelerator(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )

        # Run EKI algorithm, recording parameter error after each iteration.
        err = zeros(N_iter)
        err_acc = zeros(N_iter)
        err_acc_cs = zeros(N_iter)
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ekiobj)
            print(size(params_i))
            params_i_acc = get_ϕ_final(prior, ekiobj_acc)
            params_i_acc_cs = get_ϕ_final(prior, ekiobj_acc_cs)

            g_ens = run_G_ensemble(darcy, params_i)
            g_ens_acc = run_G_ensemble(darcy, params_i_acc)
            g_ens_acc_cs = run_G_ensemble(darcy, params_i_acc_cs)

            EKP.update_ensemble!(ekiobj, g_ens, deterministic_forward_map = false)
            EKP.update_ensemble!(ekiobj_acc, g_ens_acc, deterministic_forward_map = false)
            EKP.update_ensemble!(ekiobj_acc_cs, g_ens_acc_cs, deterministic_forward_map = false)

            err[i] = log.(get_error(ekiobj)[end])
            errs[trial, :] = err
            err_acc[i] = log.(get_error(ekiobj_acc)[end])
            errs_acc[trial, :] = err_acc
            err_acc_cs[i] = log.(get_error(ekiobj_acc_cs)[end])
            errs_acc_cs[trial, :] = err_acc_cs
        end

        GC.gc()

    end


    # compare recorded convergences with default, Nesterov, and First-Order Nesterov accelerators
    gr(size = (600, 500), legend = false)
    conv_plot = plot(
        1:N_iter,
        mean(errs, dims = 1)[:],
        ribbon = std(errs, dims = 1)[:] / sqrt(N_trials),
        color = :black,
        label = "No acceleration",
        titlefont = 20,
        legendfontsize = 13,
        guidefontsize = 15,
        tickfontsize = 15,
        linewidth = 2,
    )
    plot!(
        1:N_iter,
        mean(errs_acc, dims = 1)[:],
        ribbon = std(errs_acc, dims = 1)[:] / sqrt(N_trials),
        color = :blue,
        label = "Nesterov",
        linewidth = 2,
    )
    title!("EKI convergence on Darcy IP")
    xlabel!("Iteration")
    ylabel!("log(Cost)")
    savefig(conv_plot, joinpath(fig_save_directory, case * "_darcy_conv_comparison.png"))
    savefig(conv_plot, joinpath(fig_save_directory, case * "_darcy_conv_comparison.pdf"))
end

main()
