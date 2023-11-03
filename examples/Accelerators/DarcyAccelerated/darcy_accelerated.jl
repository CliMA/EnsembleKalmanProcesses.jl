# # [Learning the Pearmibility field in a Darcy flow from noisy sparse observations] 

# In this example we hope to illustrate function learning. One may wish to use function learning in cases where the underlying parameter of interest is actual a finite-dimensional approximation (e.g. spatial discretization) of some "true" function. Treating such an object directly will lead to increasingly high-dimensional learning problems as the spatial resolution is increased, resulting in poor computational scaling and increasingly ill-posed inverse problems. Treating the object as a discretized function from a function space, one can learn coefficients not in the standard basis, but instead in a basis of this function space, it is commonly the case that functions will have relatively low effective dimension, and will be depend only on the spatial discretization due to discretization error, that should vanish as resolution is increased. 

# We will solve for an unknown permeability field ``\kappa`` governing the pressure of a Darcy flow on a square 2D domain. To learn about the permeability we shall take few pointwise measurements of the solved pressure field within the domain. 
# The forward solver is a simple finite difference scheme taken and modified from code [here](https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl). 

# First we load standard packages
using LinearAlgebra
using Distributions
using Random
using JLD2

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

    cases = ["const", "dmc", "dmc-loc-small-ens"]
    case = cases[3]

    @info "running case $case"
    if case == "const"
        scheduler = DefaultScheduler()
        N_ens = 52 # dofs+2  
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "dmc"
        scheduler = DataMisfitController(terminate_at = 1e4)
        N_ens = 52  # dofs+2
        localization_method = EKP.Localizers.NoLocalization()
    elseif case == "dmc-loc-small-ens"
        scheduler = DataMisfitController(terminate_at = 1e4)
        N_ens = 10
        localization_method = EKP.Localizers.SEC(1.0, 0.01)
    end

    # Define the spatial domain and discretization 
    dim = 2
    N, L = 80, 1.0
    pts_per_dim = LinRange(0, L, N)
    obs_ΔN = 10

    # To provide a simple test case, we assume that the true function parameter is a particular sample from the function space we set up to define our prior. More precisely we choose a value of the truth that doesnt have a vanishingly small probability under the prior defined by a probability distribution over functions; here taken as a family of Gaussian Random Fields (GRF). The function distribution is characterized by a covariance function - here a Matern kernel which assumes a level of smoothness over the samples from the distribution. We define an appropriate expansion of this distribution, here based on the Karhunen-Loeve expansion (similar to an eigenvalue-eigenfunction expansion) that is truncated to a finite number of terms, known as the degrees of freedom (`dofs`). The `dofs` define the effective dimension of the learning problem, decoupled from the spatial discretization. Explicitly, larger `dofs` may be required to represent multiscale functions, but come at an increased dimension of the parameter space and therefore a typical increase in cost and difficulty of the learning problem.
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
    truth_sample = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))

    # Now we set up the Bayesian inversion algorithm. The prior we have already defined to construct our truth
    prior = pd

    # We define some algorithm parameters, here we take ensemble members larger than the dimension of the parameter space
    N_ens = dofs + 2    # number of ensemble members
    N_iter = 20         # number of EKI iterations
    N_trials = 10       # number of trials 
    @info "obtaining statistics over $N_trials trials"

    errs = zeros(N_trials, N_iter)
    errs_acc = zeros(N_trials, N_iter)
    errs_acc_cs = zeros(N_trials, N_iter)

    for (idx, trial) in enumerate(1:N_trials)

        @info "computing trial $idx"
        # We sample the initial ensemble from the prior, and create three EKP objects to 
        # perform EKI algorithm using three different acceleration methods.
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_params,
            truth_sample,
            obs_noise_cov,
            Inversion(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_acc = EKP.EnsembleKalmanProcess(
            initial_params,
            truth_sample,
            obs_noise_cov,
            Inversion(),
            accelerator = NesterovAccelerator(),
            scheduler = deepcopy(scheduler),
            localization_method = deepcopy(localization_method),
        )
        ekiobj_acc_cs = EKP.EnsembleKalmanProcess(
            initial_params,
            truth_sample,
            obs_noise_cov,
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
            params_i_acc = get_ϕ_final(prior, ekiobj_acc)
            params_i_acc_cs = get_ϕ_final(prior, ekiobj_acc_cs)

            g_ens = run_G_ensemble(darcy, params_i)
            g_ens_acc = run_G_ensemble(darcy, params_i_acc)
            g_ens_acc_cs = run_G_ensemble(darcy, params_i_acc_cs)

            EKP.update_ensemble!(ekiobj, g_ens, deterministic_forward_map = true)
            EKP.update_ensemble!(ekiobj_acc, g_ens_acc, deterministic_forward_map = true)
            EKP.update_ensemble!(ekiobj_acc_cs, g_ens_acc_cs, deterministic_forward_map = true)

            err[i] = log.(get_error(ekiobj)[end])
            errs[trial, :] = err
            err_acc[i] = log.(get_error(ekiobj_acc)[end])
            errs_acc[trial, :] = err_acc
            err_acc_cs[i] = log.(get_error(ekiobj_acc_cs)[end])
            errs_acc_cs[trial, :] = err_acc_cs
        end

    end


    # compare recorded convergences with default, Nesterov, and First-Order Nesterov accelerators
    gr(legend = true)
    conv_plot = plot(1:N_iter, mean((errs), dims = 1)[:], label = "No acceleration", color = "black")
    plot!(1:N_iter, mean((errs_acc), dims = 1)[:], label = "Nesterov", color = "blue")
    plot!(1:N_iter, mean((errs_acc_cs), dims = 1)[:], label = "Nesterov, FirstOrder", color = "red")
    # error bars
    plot!(
        1:N_iter,
        (mean(errs, dims = 1)[:] + std(errs, dims = 1)[:] / sqrt(N_trials)),
        color = :black,
        ls = :dash,
        label = "",
    )
    plot!(
        1:N_iter,
        (mean(errs, dims = 1)[:] - std(errs, dims = 1)[:] / sqrt(N_trials)),
        color = :black,
        ls = :dash,
        label = "",
    )
    plot!(
        1:N_iter,
        (mean(errs_acc, dims = 1)[:] + std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
        color = :blue,
        ls = :dash,
        label = "",
    )
    plot!(
        1:N_iter,
        (mean(errs_acc, dims = 1)[:] - std(errs_acc, dims = 1)[:] / sqrt(N_trials)),
        color = :blue,
        ls = :dash,
        label = "",
    )
    plot!(
        1:N_iter,
        (mean(errs_acc_cs, dims = 1)[:] + std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
        color = :red,
        ls = :dash,
        label = "",
    )
    plot!(
        1:N_iter,
        (mean(errs_acc_cs, dims = 1)[:] - std(errs_acc_cs, dims = 1)[:] / sqrt(N_trials)),
        color = :red,
        ls = :dash,
        label = "",
    )

    title!("EKI convergence on Darcy IP")
    xlabel!("Iteration")
    ylabel!("log(Error)")
    savefig(conv_plot, joinpath(fig_save_directory, case * "_darcy_conv_comparison.png"))
end

main()
