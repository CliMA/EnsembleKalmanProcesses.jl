##################
# Code snippets taken from 3/16/23 from 
# https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl
##################

using LinearAlgebra
using Distributions
using Random
import GaussianRandomFields # we wrap this so we don't want to use "using"
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const GRF = GaussianRandomFields
const EKP = EnsembleKalmanProcesses
include("GModel.jl")
using JLD2

PLOT_FLAG = true

fig_save_directory = joinpath(@__DIR__, "output")
data_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end

if PLOT_FLAG
    using Plots
    @info "Plotting enabled, this will reduce code performance. Figures stored in $fig_save_directory"
end

seed = 100234
rng = Random.MersenneTwister(seed)


function main()

    #Configuration
    N, L = 80, 1.0
    xx = LinRange(0, L, N)
    obs_ΔN = 10

    # Set up the family of function distributions and create the truth with it)
    dim = 2
    smoothness = 2.0
    corr_length = 0.5
    dofs = 128

    grf = GRF.GaussianRandomField(
        GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
        GRF.KarhunenLoeve(dofs),
        xx,
        xx,
    ) # the constructor from GRF

    pkg = GRFJL()
    distribution = GaussianRandomFieldInterface(grf, pkg) # our wrapper from EKP

    domain_constraint = bounded_below(0) # make κ positive
    pd = ParameterDistribution(
        Dict("distribution" => distribution, "name" => "kappa", "constraint" => domain_constraint),
    ) # the fully constrained parameter distribution

    # sample the distribution directly with a given coefficient for the truth
    u_true = rand(rng, Uniform(0.4, 0.5), dofs) # the truth parameter
    println("True coefficients: ")
    println(u_true)
    κ_true = transform_unconstrained_to_constrained(pd, u_true) # builds and constrains the function.  
    κ_true = reshape(κ_true, N, N)

    darcy = Setup_Param(xx, obs_ΔN, κ_true)

    h_2d = solve_Darcy_2D(darcy, κ_true)
    y_noiseless = compute_obs(darcy, h_2d)

    if PLOT_FLAG
        gr(size = (1000, 400), legend = false)
        p1 = contour(xx, xx, κ_true', fill = true, levels = 15, title = "kappa true", colorbar = true)
        p2 = contour(xx, xx, h_2d', fill = true, levels = 15, title = "flow true", colorbar = true)
        l = @layout [a b]
        plt = plot(p1, p2, layout = l)
        savefig(plt, joinpath(fig_save_directory, "output_true.png"))
    end

    # data
    obs_noise_cov = 0.05^2 * I(length(y_noiseless))
    truth_sample = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))


    # now setup the Bayesian inversion algorithm
    prior = pd # NB, `distribution` internally codes a default prior on the coefficients

    # EKP parameters
    N_ens = dofs + 2 # number of ensemble members
    N_iter = 10 # number of EKI iterations
    # initial parameters: N_params x N_ens
    initial_params = construct_initial_ensemble(rng, prior, N_ens)
    ekiobj = EKP.EnsembleKalmanProcess(initial_params, truth_sample, obs_noise_cov, Inversion())

    println("Begin inversion")
    err = zeros(N_iter)
    for i in 1:N_iter
        params_i = get_ϕ_final(prior, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space
        if PLOT_FLAG
            gr(size = (1500, 400), legend = false)
            κ_ens_mean = reshape(mean(params_i, dims = 2), N, N)
            p1 = contour(xx, xx, κ_ens_mean', fill = true, levels = 15, title = "kappa mean", colorbar = true)
            κ_ens_ptw_var = reshape(var(params_i, dims = 2), N, N)
            p2 = contour(xx, xx, κ_ens_ptw_var', fill = true, levels = 15, title = "kappa var", colorbar = true)
            h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
            p3 = contour(xx, xx, h_2d', fill = true, levels = 15, title = "flow", colorbar = true)
            l = @layout [a b c]
            plt = plot(p1, p2, p3, layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_it_" * string(i - 1) * ".png")) # pre update

        end
        g_ens = run_G_ensemble(darcy, params_i)


        EKP.update_ensemble!(ekiobj, g_ens)
        err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
        println("Iteration: " * string(i) * ", Error: " * string(err[i]))
    end

    if PLOT_FLAG
        gr(size = (1500, 400), legend = false)
        final_κ_ens = get_ϕ_final(prior, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space
        κ_ens_mean = reshape(mean(final_κ_ens, dims = 2), N, N)
        p1 = contour(xx, xx, κ_ens_mean', fill = true, levels = 15, title = "kappa mean", colorbar = true)
        κ_ens_ptw_var = reshape(var(final_κ_ens, dims = 2), N, N)
        p2 = contour(xx, xx, κ_ens_ptw_var', fill = true, levels = 15, title = "kappa var", colorbar = true)
        h_2d = solve_Darcy_2D(darcy, κ_ens_mean)
        p3 = contour(xx, xx, h_2d', fill = true, levels = 15, title = "flow", colorbar = true)
        l = @layout [a b c]
        plt = plot(p1, p2, p3; layout = l)
        savefig(plt, joinpath(fig_save_directory, "output_it_" * string(N_iter) * ".png")) # pre update

    end
    println("Final coefficients (ensemble mean):")
    println(get_u_mean_final(ekiobj))


    u_stored = get_u(ekiobj, return_array = false)
    g_stored = get_g(ekiobj, return_array = false)
    @save joinpath(data_save_directory, "parameter_storage.jld2") u_stored
    @save joinpath(data_save_directory, "data_storage.jld2") g_stored





end

main()
