# Import modules
include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots

using Random
using JLD2

using Plots
# CES 
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

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
# Lognormal prior or normal prior?
log_normal = false # THIS ISN't CURRENTLY IMPLEMENTED

function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2 / μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2 / μ^2)))
    return μ_log, σ_log
end
#logmean_F, logstd_F = logmean_and_logstd(F_true, 5)
#logmean_A, logstd_A = logmean_and_logstd(A_true, 0.2*A_true)

if dynamics == 2
    prior_mean = [F_true + 1.0; A_true + 0.5]
    # prior_stds = [2.0, 0.5*A_true]
    prior_cov = Array(Diagonal((0.1 * prior_mean) .^ 2))
else
    prior_mean = [F_true;]
    prior_cov = ones(Float64, 1, 1)
end


###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0", "y1"]


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
truth = Observations.Obs(yt, Γy, data_names)
truth_sample = truth.mean
###
###  Calibrate: Ensemble Kalman Inversion
###

# L96 settings for the forward model in the EKP
# Here, the forward model for the EKP settings can be set distinctly from the truth runs
lorenz_settings_G = lorenz_settings; # initialize to truth settings

# EKP parameters
log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)


N_iter = 20 # number of UKI iterations
# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg = 1.0

# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 0

process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
ukiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)


# UKI iterations
err = zeros(N_iter)
for i in 1:N_iter
    if log_normal == false
        params_i = get_u_final(ukiobj)
    else
        params_i = exp_transform(get_u_final(ukiobj))
    end
    g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)


    # analysis step 
    EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens)

    err[i] = get_error(ukiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println(
        "Iteration: " *
        string(i) *
        ", Error: " *
        string(err[i]) *
        " norm(Cov): " *
        string(norm(ukiobj.process.uu_cov[i])),
    )
end



# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nUKI results:")
if log_normal == false
    println(get_u_mean_final(ukiobj))
else
    println(exp_transform(get_u_mean_final(ukiobj)))
end




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
plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "F")
plot!(ites, fill(params_true[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)

plot!(ites, grid = false, θ_mean_arr[2, :], yerror = 3.0 * θθ_std_arr[2, :], label = "A", xaxis = "Iterations")
plot!(ites, fill(params_true[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)



#plots
gr(size = (600, 600))
dθ = 1.0
for i in 1:N_iter
    θ_mean, θθ_cov = ukiobj.process.u_mean[i], ukiobj.process.uu_cov[i]
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
    display(p)
    sleep(0.5)
end
