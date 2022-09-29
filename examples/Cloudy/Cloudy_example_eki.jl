# This example requires Cloudy to be installed (it's best to install the master
# branch), which can be done by:
#] add Cloudy#master
using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.KernelTensors

# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using Plots
using Plots.PlotMeasures
using JLD2
using Random

# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.DataContainers

# Import the module that runs Cloudy
include("DynamicalModel.jl")
using .DynamicalModel

rng_seed = 41
Random.seed!(rng_seed)

homedir = pwd()
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

###
###  Define the (true) parameters and their priors
###

# Define the parameters that we want to learn
# We assume that the true particle mass distribution is a Gamma 
# distribution with parameters N0_true, θ_true, k_true
par_names = ["N0", "θ", "k"]
n_par = length(par_names)
N0_true = 300.0  # number of particles (scaling factor for Gamma distribution)
θ_true = 1.5597  # scale parameter of Gamma distribution
k_true = 0.0817  # shape parameter of Gamma distribution
# Note that dist_true is a Cloudy distribution, not a Distributions.jl 
# distribution
ϕ_true = [N0_true, θ_true, k_true]
dist_true = ParticleDistributions.GammaPrimitiveParticleDistribution(ϕ_true...)


###
###  Define priors for the parameters we want to learn
###
# We choose to use normal distributions to represent the prior distributions of
# the parameters in the transformed (unconstrained) space. i.e log coordinates
prior_N0 = constrained_gaussian(par_names[1], 400, 300, 0.4 * N0_true, Inf)
prior_θ = constrained_gaussian(par_names[2], 1.0, 5.0, 1e-1, Inf)
prior_k = constrained_gaussian(par_names[3], 0.2, 1.0, 1e-4, Inf)
priors = combine_distributions([prior_N0, prior_θ, prior_k])

###
###  Define the data from which we want to learn the parameters
###

data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)


###
###  Model settings
###

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1 / 3.14 / 4 / 100
kernel_func = x -> coalescence_coeff
kernel = CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0.0, 1.0)


###
###  Generate (artificial) truth samples
###

model_settings_true = ModelSettings(kernel, dist_true, moments, tspan)
G_t = run_dyn_model(ϕ_true, model_settings_true)
n_samples = 100
y_t = zeros(length(G_t), n_samples)
# In a perfect model setting, the "observational noise" represents the 
# internal model variability. Since Cloudy is a purely deterministic model, 
# there is no straightforward way of coming up with a covariance structure 
# for this internal model variability. We decide to use a diagonal 
# covariance, with entries (variances) largely proportional to their 
# corresponding data values, G_t
Γy = convert(Array, Diagonal([100.0, 5.0, 30.0]))
μ = zeros(length(G_t))

# Add noise
for i in 1:n_samples
    y_t[:, i] = G_t .+ rand(MvNormal(μ, Γy))
end

truth = Observations.Observation(y_t, Γy, data_names)
truth_sample = truth.mean


###
###  Calibrate: Ensemble Kalman Inversion
###

N_ens = 50 # number of ensemble members
N_iter = 8 # number of EKI iterations
# initial parameters: N_par x N_ens
initial_par = construct_initial_ensemble(priors, N_ens; rng_seed)
ekiobj = EnsembleKalmanProcess(initial_par, truth_sample, truth.obs_noise_cov, Inversion(), Δt = 0.1)

# Initialize a ParticleDistribution with dummy parameters. The parameters 
# will then be set within `run_dyn_model`
dummy = ones(n_par)
dist_type = ParticleDistributions.GammaPrimitiveParticleDistribution(dummy...)
model_settings = DynamicalModel.ModelSettings(kernel, dist_type, moments, tspan)
# EKI iterations
for n in 1:N_iter
    # Return transformed parameters in physical/constrained space
    ϕ_n = get_ϕ_final(priors, ekiobj)
    # Evaluate forward map
    G_n = [run_dyn_model(ϕ_n[:, i], model_settings) for i in 1:N_ens]
    G_ens = hcat(G_n...)  # reformat
    EnsembleKalmanProcesses.update_ensemble!(ekiobj, G_ens)
end

# EKI results: Has the ensemble collapsed toward the truth?
θ_true = transform_constrained_to_unconstrained(priors, ϕ_true)
println("True parameters (unconstrained): ")
println(θ_true)

println("\nEKI results:")
println(get_u_mean_final(ekiobj))

u_stored = get_u(ekiobj, return_array = false)
g_stored = get_g(ekiobj, return_array = false)
@save data_save_directory * "parameter_storage_eki.jld2" u_stored
@save data_save_directory * "data_storage_eki.jld2" g_stored

#plots - unconstrained
gr(size = (1200, 400))

u_init = get_u_prior(ekiobj)
anim_eki_unconst_cloudy = @animate for i in 1:N_iter
    u_i = get_u(ekiobj, i)

    p1 = plot(u_i[1, :], u_i[2, :], seriestype = :scatter, xlims = extrema(u_init[1, :]), ylims = extrema(u_init[2, :]))
    plot!(
        p1,
        [θ_true[1]],
        xaxis = "u1",
        yaxis = "u2",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        margin = 5mm,
        title = "EKI iteration = " * string(i),
    )
    plot!(p1, [θ_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p2 = plot(u_i[2, :], u_i[3, :], seriestype = :scatter, xlims = extrema(u_init[2, :]), ylims = extrema(u_init[3, :]))
    plot!(
        p2,
        [θ_true[2]],
        xaxis = "u2",
        yaxis = "u3",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        margin = 5mm,
        title = "EKI iteration = " * string(i),
    )
    plot!(p2, [θ_true[3]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p3 = plot(u_i[3, :], u_i[1, :], seriestype = :scatter, xlims = extrema(u_init[3, :]), ylims = extrema(u_init[1, :]))
    plot!(
        p3,
        [θ_true[3]],
        xaxis = "u3",
        yaxis = "u1",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        margin = 5mm,
        title = "EKI iteration = " * string(i),
    )
    plot!(p3, [θ_true[1]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p = plot(p1, p2, p3, layout = (1, 3))
end
gif(anim_eki_unconst_cloudy, joinpath(figure_save_directory, "eki_unconst_cloudy.gif"), fps = 1) # hide

# plots - constrained
ϕ_init = transform_unconstrained_to_constrained(priors, u_init)
anim_eki_cloudy = @animate for i in 1:N_iter
    ϕ_i = get_ϕ(priors, ekiobj, i)

    p1 = plot(ϕ_i[1, :], ϕ_i[2, :], seriestype = :scatter, xlims = extrema(ϕ_init[1, :]), ylims = extrema(ϕ_init[2, :]))
    plot!(
        p1,
        [ϕ_true[1]],
        xaxis = "ϕ1",
        yaxis = "ϕ2",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        margin = 5mm,
        label = false,
        title = "EKI iteration = " * string(i),
    )
    plot!(p1, [ϕ_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p2 = plot(ϕ_i[2, :], ϕ_i[3, :], seriestype = :scatter, xlims = extrema(ϕ_init[2, :]), ylims = extrema(ϕ_init[3, :]))
    plot!(
        p2,
        [ϕ_true[2]],
        xaxis = "ϕ2",
        yaxis = "ϕ3",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        margin = 5mm,
        label = false,
        title = "EKI iteration = " * string(i),
    )
    plot!(p2, [ϕ_true[3]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p3 = plot(ϕ_i[3, :], ϕ_i[1, :], seriestype = :scatter, xlims = extrema(ϕ_init[3, :]), ylims = extrema(ϕ_init[1, :]))
    plot!(
        p3,
        [ϕ_true[3]],
        xaxis = "ϕ3",
        yaxis = "ϕ1",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        margin = 5mm,
        label = false,
        title = "EKI iteration = " * string(i),
    )
    plot!(p3, [ϕ_true[1]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p = plot(p1, p2, p3, layout = (1, 3))
end
gif(anim_eki_cloudy, joinpath(figure_save_directory, "eki_cloudy.gif"), fps = 1) # hide
