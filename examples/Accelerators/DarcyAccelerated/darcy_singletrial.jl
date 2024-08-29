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
    N_trials = 5       # number of trials 

    N_iter = 10 # number of EKI iterations
    initial_params = construct_initial_ensemble(rng, prior, N_ens)
    eki_trad = EKP.EnsembleKalmanProcess(
        initial_params,
        truth_sample,
        obs_noise_cov,
        Inversion(),
        accelerator = DefaultAccelerator(),
        scheduler = DefaultScheduler(0.1),
    )#,scheduler = DataMisfitController(on_terminate = "continue"))
    eki_acc = EKP.EnsembleKalmanProcess(
        initial_params,
        truth_sample,
        obs_noise_cov,
        Inversion(),
        accelerator = NesterovAccelerator(),
        scheduler = DefaultScheduler(0.1),
    )

    err = zeros(N_iter)
    err_acc = zeros(N_iter)

    for i in 1:N_iter
        params_i = get_ϕ_final(prior, eki_trad)
        params_i_acc = get_ϕ_final(prior, eki_acc)

        g_ens = run_G_ensemble(darcy, params_i)
        g_ens_acc = run_G_ensemble(darcy, params_i_acc)

        EKP.update_ensemble!(eki_trad, g_ens, deterministic_forward_map = false)
        EKP.update_ensemble!(eki_acc, g_ens_acc)

        err[i] = get_error(eki_trad)[end]    # mean((params_true - mean(params_i,dims=2)).^2)
        err_acc[i] = get_error(eki_acc)[end]    # mean((params_true - mean(params_i,dims=2)).^2)
    end
    println("Error, traditional EKI")
    println(err)
    println("Error, accelerated EKI")
    println(err_acc)

    ## Now we plot the final ensemble mean and pointwise variance of the permeability field, and also the pressure field solved with the ensemble mean.
    h_2d_true = solve_Darcy_2D(darcy, κ_true)

    gr(size = (1500, 400), legend = false)
    final_κ_ens = get_ϕ_final(prior, eki_trad) # the `ϕ` indicates that the `params_i` are in the constrained space
    κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
    p1 = contour(
        pts_per_dim,
        pts_per_dim,
        κ_ens_mean',
        fill = true,
        levels = 15,
        title = "kappa_mean (traditional EKI)",
        colorbar = true,
    )
    κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
    h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
    p3 = contour(
        pts_per_dim,
        pts_per_dim,
        h_2d',
        fill = true,
        levels = 15,
        title = "pressure (traditional EKI)",
        colorbar = true,
    )
    l = @layout [a c]
    plt = plot(p1, p3; layout = l)
    savefig(plt, joinpath(fig_save_directory, "output_it_" * string(N_iter) * ".png"))

    gr(size = (1500, 400), legend = false)
    final_κ_ens = get_ϕ_final(prior, eki_acc) # the `ϕ` indicates that the `params_i` are in the constrained space
    κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
    p1 = contour(
        pts_per_dim,
        pts_per_dim,
        κ_ens_mean',
        fill = true,
        levels = 15,
        title = "kappa_mean (EKI with momentum)",
        colorbar = true,
    )
    κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
    h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
    p3 = contour(
        pts_per_dim,
        pts_per_dim,
        h_2d',
        fill = true,
        levels = 15,
        title = "pressure (EKI with momentum)",
        colorbar = true,
    )
    l = @layout [a c]
    plt = plot(p1, p3; layout = l)
    savefig(plt, joinpath(fig_save_directory, "output_it_" * string(N_iter) * "_acc.png"))

    gr(size = (1500, 400), legend = false)
    final_κ_ens = get_ϕ_final(prior, eki_trad) # the `ϕ` indicates that the `params_i` are in the constrained space
    κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
    p1 = contour(
        pts_per_dim,
        pts_per_dim,
        (κ_ens_mean .- κ_true)',
        fill = true,
        levels = 20,
        title = "kappa error (traditional EKI)",
        colorbar = true,
        clims = (-30, 30),
    )
    κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
    h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
    p3 = contour(
        pts_per_dim,
        pts_per_dim,
        (h_2d .- h_2d_true)',
        fill = true,
        levels = 20,
        title = "pressure error (traditional EKI); error = " * string(floor(err[end])),
        colorbar = true,
        clims = (-45, 45),
    )
    l = @layout [a c]
    plt = plot(p1, p3; layout = l)
    savefig(plt, joinpath(fig_save_directory, "output_diff_it_" * string(N_iter) * ".png"))

    gr(size = (1500, 400), legend = false)
    final_κ_ens = get_ϕ_final(prior, eki_acc) # the `ϕ` indicates that the `params_i` are in the constrained space
    κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
    p1 = contour(
        pts_per_dim,
        pts_per_dim,
        (κ_ens_mean .- κ_true)',
        fill = true,
        levels = 20,
        title = "kappa error (EKI with momentum)",
        colorbar = true,
        clims = (-30, 30),
    )
    κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
    h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
    p3 = contour(
        pts_per_dim,
        pts_per_dim,
        (h_2d .- h_2d_true)',
        fill = true,
        levels = 20,
        title = "pressure error (EKI with momentum); error = " * string(floor(err_acc[end])),
        colorbar = true,
        clims = (-45, 45),
    )
    l = @layout [a c]
    plt = plot(p1, p3; layout = l)
    savefig(plt, joinpath(fig_save_directory, "output_diff_it_" * string(N_iter) * "_acc.png"))

    h_2d = solve_Darcy_2D(darcy, κ_true) #?
    gr(size = (1500, 400), legend = false)
    p1 = contour(pts_per_dim, pts_per_dim, κ_true', fill = true, levels = 15, title = "kappa true", colorbar = true)
    p2 = contour(pts_per_dim, pts_per_dim, h_2d', fill = true, levels = 15, title = "pressure true", colorbar = true)
    l = @layout [a b]
    plt = plot(p1, p2, layout = l)
    savefig(plt, joinpath(fig_save_directory, "output_true.png"))
end

main()
