include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

const EKP = EnsembleKalmanProcesses

#######
#Running the Lorenz model:
# Solve the Lorenz 96 system 
# Inputs: F: scalar forcing Float64(1), Fp: initial perturbation Float64(N)
# N: number of longitude steps Int64(1), dt: time step Float64(1)
# tend: End of simulation Float64(1), nstep:

G(θ) = H(Ψ(θ,x₀,t₀,t₁))
y = G(θ) + η

# This will change for different Lorenz simulators
struct LorenzConfig{FT <: Real}
    dt::FT
    T::FT
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    "state-dependent-forcing"
    F::VV
end

# This will change for different "Observations" of Lorenz
struct ObservationConfig{FT <: Real} 
    t_start::FT
    t_end::FT
end



function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig , ) <: {VorM <: AbstractVecOrMat}
    # Initialize    
    nstep = Int(ceil(config.T / config.dt))
    state_dim = isa(x0,AbstractVector) ? length(x0) : size(x0,1)
    xn = zeros(size(x0,1), nstep+1)
    xn[:,1] = x0
    
    # March forward in time
    for j in 1:nstep
        xn[:,j+1] = RK4(params, xn[:,j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: x: state, N: state dimension, F: forcing
function f(params, x)
    F = params.F
    N = length(x)
    f = zeros(N)
    # Loop over N positions
    for i in 3:(N - 1)
        f[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + F[i]
    end
    # Periodic boundary conditions
    f[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] + F[1]
    f[2] = -x[N] * x[1] + x[1] * x[3] - x[2] + F[2]
    f[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] + F[N]
    # Output
    return f
end

# RK4 solve
function RK4(params, xold, config)
    N = length(xold)
    dt = config.dt

    # Predictor steps (note no time-dependence is needed here)
    k1 = f(params, xold)
    k2 = f(params, xold + k1 * dt / 2.0)
    k3 = f(params, xold + k2 * dt / 2.0)
    k4 = f(params, xold + k3 * dt)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end







rng_seed = 4137
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

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
dynamics = 1 # This fixes the forcing to be stationary in time.
# Statistics integration length
# This has to be less than 360 and 360 must be divisible by Ts_days
Ts_days = 90.0 # Integration length in days
# Stats type, which statistics to construct from the L96 system
# 4 is a linear fit over a batch of length Ts_days
# 5 is the mean over a batch of length Ts_days
stats_type = 5

































