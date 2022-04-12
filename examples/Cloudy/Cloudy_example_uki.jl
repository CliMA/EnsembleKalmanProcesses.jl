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
using JLD2
using Random

# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

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

# N0
lbound_N0 = 0.4 * N0_true
prior_N0 = Dict(
    "distribution" => Parameterized(Normal(4.5, 1.0)), #truth is 5.19
    "constraint" => bounded_below(lbound_N0),
    "name" => par_names[1],
)

# θ
lbound_θ = 1.0e-1
prior_θ = Dict(
    "distribution" => Parameterized(Normal(0.0, 2.0)),  #truth is 0.378
    "constraint" => bounded_below(lbound_θ),
    "name" => par_names[2],
)


# k
lbound_k = 1.0e-4
prior_k = Dict(
    "distribution" => Parameterized(Normal(-1.0, 1.0)), #truth is -2.51 
    "constraint" => bounded_below(lbound_k),
    "name" => par_names[3],
)

priors = ParameterDistribution([prior_N0, prior_θ, prior_k])
prior_mean = [4.5; 0.0; -1.0]
prior_cov = Array(Diagonal(ones(Float64, length(prior_mean))))


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
###  Calibrate: Unscented Kalman Inversion
###

N_iter = 20 # number of iterations
# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg = 1.0
# update_freq 1 : approximate posterior covariance matrix with an uninformative
#                 prior
#             0 : weighted average between posterior covariance matrix with an
#                 uninformative prior and prior
update_freq = 1

process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

# Initialize a ParticleDistribution with dummy parameters. The parameters 
# will then be set within `run_dyn_model`
dummy = ones(n_par)
dist_type = ParticleDistributions.GammaPrimitiveParticleDistribution(dummy...)
model_settings = DynamicalModel.ModelSettings(kernel, dist_type, moments, tspan)

err = zeros(N_iter)
for n in 1:N_iter
    θ_n = get_u_final(ukiobj)
    # Transform parameters to physical/constrained space
    ϕ_n = mapslices(x -> transform_unconstrained_to_constrained(priors, x), θ_n; dims = 1)
    # Evaluate forward map
    println("size: ", size(ϕ_n))
    G_n = [run_dyn_model(ϕ_n[:, i], model_settings) for i in 1:size(ϕ_n)[2]]
    G_ens = hcat(G_n...)  # reformat
    EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
    err[n] = get_error(ukiobj)[end]
    println(
        "Iteration: " *
        string(n) *
        ", Error: " *
        string(err[n]) *
        " norm(Cov):
" *
        string(norm(ukiobj.process.uu_cov[n])),
    )
end


# UKI results: the mean is in ukiobj.process.u_mean
#              the covariance matrix is in ukiobj.process.uu_cov
θ_true = transform_constrained_to_unconstrained(priors, ϕ_true)

println("True parameters (transformed): ")
println(θ_true)

println("\nUKI results:")
println(get_u_mean_final(ukiobj))

u_stored = get_u(ukiobj, return_array = false)
g_stored = get_g(ukiobj, return_array = false)
@save data_save_directory * "parameter_storage_uki.jld2" u_stored
@save data_save_directory * "data_storage_uki.jld2" g_stored

####
θ_mean_arr = hcat(ukiobj.process.u_mean...)
N_θ = length(ukiobj.process.u_mean[1])
θθ_std_arr = zeros(Float64, (N_θ, N_iter + 1))
for i in 1:(N_iter + 1)
    for j in 1:N_θ
        θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j, j])
    end
end


ites = Array(LinRange(1, N_iter + 1, N_iter + 1))
plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "u1")
plot!(ites, fill(θ_true[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)
plot!(ites, grid = false, θ_mean_arr[2, :], yerror = 3.0 * θθ_std_arr[2, :], label = "u2", xaxis = "Iterations")
plot!(ites, fill(θ_true[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)
plot!(ites, grid = false, θ_mean_arr[3, :], yerror = 3.0 * θθ_std_arr[3, :], label = "u3", xaxis = "Iterations")
plot!(ites, fill(θ_true[3], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)

gr(size = (1800, 600))

for i in 1:N_iter
    θ_mean, θθ_cov = ukiobj.process.u_mean[i], ukiobj.process.uu_cov[i]
    θ1, θ2, fθ1θ2 = Gaussian_2d(θ_mean[1:2], θθ_cov[1:2, 1:2], 100, 100)
    p1 = contour(θ1, θ2, fθ1θ2)
    plot!(
        p1,
        [θ_true[1]],
        xaxis = "u1",
        yaxis = "u2",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "UKI iteration = " * string(i),
    )
    plot!(p1, [θ_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    θ2, θ3, fθ2θ3 = Gaussian_2d(θ_mean[2:3], θθ_cov[2:3, 2:3], 100, 100)
    p2 = contour(θ2, θ3, fθ2θ3)
    plot!(
        p2,
        [θ_true[2]],
        xaxis = "u2",
        yaxis = "u3",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "UKI iteration = " * string(i),
    )
    plot!(p2, [θ_true[3]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    θ3, θ1, fθ3θ1 = Gaussian_2d(θ_mean[3:-2:1], θθ_cov[3:-2:1, 3:-2:1], 100, 100)
    p3 = contour(θ3, θ1, fθ3θ1)
    plot!(
        p3,
        [θ_true[3]],
        xaxis = "u3",
        yaxis = "u1",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "UKI iteration = " * string(i),
    )
    plot!(p3, [θ_true[1]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")

    p = plot(p1, p2, p3, layout = (1, 3))
    display(p)
    sleep(0.5)
end
