# Import modules
include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots

using Random
using JLD2

using Plots
# EKP 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

rng_seed = 4137
Random.seed!(rng_seed)

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

# Governing settings
# Characteristic time scale
τc = 5.0 # days, prescribed by the L96 problem
# Stationary or transient dynamics
dynamics = 2 # Transient is 2
# Statistics integration length
# This has to be less than 360 and 360 must be divisible by Ts_days
Ts_days = 90.0 # Integration length in days
# Stats type, which statistics to construct from the L96 system
# 4 is a linear fit over a batch of length Ts_days
# 5 is the mean over a batch of length Ts_days
stats_type = 5


###
###  Define the (true) parameters
###
# Define the parameters that we want to learn
F_true = 8.0 # Mean F
A_true = 2.5 # Transient F amplitude
ω_true = 2.0 * π / (360.0 / τc) # Frequency of the transient F

if dynamics == 2
    params_true = [F_true, A_true]
    param_names = ["F", "A"]
else
    params_true = [F_true]
    param_names = ["F"]
end
n_param = length(param_names)
params_true = reshape(params_true, (n_param, 1))

println(n_param)
println(params_true)


###
###  Define the parameter priors
###
if dynamics == 2
    prior_means = [F_true + 1.0, A_true + 0.5]
    prior_stds = [2.0, 0.5 * A_true]
    prior_F = constrained_gaussian(param_names[1], prior_means[1], prior_stds[1], -Inf, Inf)
    prior_A = constrained_gaussian(param_names[2], prior_means[2], prior_stds[2], -Inf, Inf)
    priors = combine_distributions([prior_F, prior_A])
else
    priors = constrained_gaussian("F", F_true, 1.0, -Inf, Inf)
end

###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0_y1"]


###
###  L96 model settings
###

# Lorenz 96 model parameters
# Behavior changes depending on the size of F, N
N = 36
dt = 1 / 64.0
# Start of integration
t_start = 800.0
# Data collection length
if dynamics == 1
    T = 2.0
else
    T = 360.0 / τc
end
# Batch length
Ts = 5.0 / τc # Nondimensionalize by L96 timescale
# Integration length
Tfit = Ts_days / τc
# Initial perturbation
Fp = rand(Normal(0.0, 0.01), N);
kmax = 1
# Prescribe variance or use a number of forward passes to define true interval variability
var_prescribe = false
# Use CES to learn ω?
ω_fixed = true

# Settings
# Constructs an LSettings structure, see GModel.jl for the descriptions
lorenz_settings =
    GModel.LSettings(dynamics, stats_type, t_start, T, Ts, Tfit, Fp, N, dt, t_start + T, kmax, ω_fixed, ω_true);
lorenz_params = GModel.LParams(F_true, ω_true, A_true)

###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(θ) + η
###

# Lorenz forward
# Input: params: [N_params, N_ens]
# Output: gt: [N_data, N_ens]
# Dropdims of the output since the forward model is only being run with N_ens=1 
# corresponding to the truth construction
gt = dropdims(GModel.run_G_ensemble(params_true, lorenz_settings), dims = 2)

# Compute internal variability covariance
if var_prescribe == true
    n_samples = 100
    yt = zeros(length(gt), n_samples)
    noise_level = 0.05
    Γy = noise_level * convert(Array, Diagonal(gt))
    μ = zeros(length(gt))
    # Add noise
    for i in 1:n_samples
        yt[:, i] = gt .+ rand(MvNormal(μ, Γy))
    end
else
    println("Using truth values to compute covariance")
    n_samples = 20
    yt = zeros(length(gt), n_samples)
    for i in 1:n_samples
        lorenz_settings_local = GModel.LSettings(
            dynamics,
            stats_type,
            t_start + T * (i - 1),
            T,
            Ts,
            Tfit,
            Fp,
            N,
            dt,
            t_start + T * (i - 1) + T,
            kmax,
            ω_fixed,
            ω_true,
        )
        yt[:, i] = GModel.run_G_ensemble(params_true, lorenz_settings_local)
    end
    # Variance of truth data
    #Γy = convert(Array, Diagonal(dropdims(mean((yt.-mean(yt,dims=1)).^2,dims=1),dims=1)))
    # Covariance of truth data
    Γy = cov(yt, dims = 2)

    println(Γy)
end

# Construct observation object
truth = Observation(Dict("samples" => vec(mean(yt, dims = 2)), "covariances" => Γy, "names" => data_names))
###
###  Calibrate: Ensemble Kalman Inversion
###

# L96 settings for the forward model in the EKP
# Here, the forward model for the EKP settings can be set distinctly from the truth runs
lorenz_settings_G = lorenz_settings; # initialize to truth settings

# EKP parameters
N_iter = 20 # number of UKI iterations
# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg = 1.0

# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 0

# N_ens 2n_param+1 : sample 2n_param+1 sigma points
#       n_param+2  : sample n_param+2 sigma points
N_ens = 2n_param + 1
process = Unscented(mean(priors), cov(priors); α_reg = α_reg, update_freq = update_freq, sigma_points = "symmetric")
ukiobj = EKP.EnsembleKalmanProcess(truth, process)


# UKI iterations
err = zeros(N_iter)
for i in 1:N_iter
    params_i = get_ϕ_final(priors, ukiobj)
    g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)
    EKP.update_ensemble!(ukiobj, g_ens)

    err[i] = get_error(ukiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println(
        "Iteration: " *
        string(i) *
        ", Error: " *
        string(err[i]) *
        " norm(Cov): " *
        string(norm(get_process(ukiobj).uu_cov[i])),
    )
end


# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nUKI results:")
println(get_ϕ_mean_final(priors, ukiobj))

#### - stats in unconstrained space
θ_mean_arr = hcat([get_u_mean(ukiobj, i) for i in 1:N_iter]...)
N_θ = length(get_u_mean(ukiobj, 1))
θθ_std_arr = zeros(Float64, (N_θ, N_iter + 1))
for i in 1:(N_iter + 1)
    for j in 1:N_θ
        θθ_std_arr[j, i] = sqrt(get_u_cov(ukiobj, i)[j, j])
    end
end

ites = Array(LinRange(1, N_iter + 1, N_iter + 1))
plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "F")
plot!(ites, fill(params_true[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)

plot!(ites, grid = false, θ_mean_arr[2, :], yerror = 3.0 * θθ_std_arr[2, :], label = "A", xaxis = "Iterations")
plot!(ites, fill(params_true[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)



#plots - unconstrained space
gr(size = (600, 600))
dθ = 1.0
anim_ukp_lorenz = @animate for i in 1:N_iter
    θ_mean, θθ_cov = get_u_mean(ukiobj, i), get_u_cov(ukiobj, i)
    xx = Array(LinRange(params_true[1] - dθ, params_true[1] + dθ, 100))
    yy = Array(LinRange(params_true[2] - dθ, params_true[2] + dθ, 100))
    xx, yy, Z = Gaussian_2d(θ_mean, θθ_cov, 100, 100, xx = xx, yy = yy)

    p = contour(xx, yy, Z)
    plot!(
        [params_true[1]],
        xaxis = "F",
        yaxis = "A",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "UKI iteration = " * string(i),
    )
    plot!([params_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")
end

gif(anim_ukp_lorenz, joinpath(figure_save_directory, "ukp_unconstrained_lorenz.gif"), fps = 1) # hide
