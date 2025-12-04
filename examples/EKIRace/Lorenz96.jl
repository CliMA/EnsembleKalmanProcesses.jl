# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics
using Flux

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    val::VV
end

# Sub-type of ensemble config for constant forcing
struct ConstantEMC{FT <: Real} <: EnsembleMemberConfig
   val::FT
end
build_forcing(val::FT, args...) where {FT <: Real} = ConstantEMC(val[1])

# Sub-type of ensemble config for spatially-dependent forcing
struct VectorEMC{VV<:AbstractVector} <: EnsembleMemberConfig
   val::VV
end
build_forcing(val::FT, args...) where {FT <: AbstractVector} = VectorEMC(val)

# Sub-type of ensemble config for spatially-dependent forcing with neural network approximation
struct FluxEMC <: EnsembleMemberConfig
    model::Flux.Chain
end
function build_forcing(model,params)
    _ , reconstructor = Flux.destructure(model)
    return FluxEMC(reconstructor(params))
end

# Constant-global
forcing(params::ConstantEMC, x, i) = params.val
    
# Constant-vector
forcing(params::VectorEMC, x, i) = params.val[i]
        
# Flux
forcing(params::FluxEMC, x, i) = Float64(params.model([Float32(i)])[1])




# This will change for different "Observations" of Lorenz
struct ObservationConfig{FT1 <: Real, FT2 <: Real}
    "initial time to gather statistics (T_start = N_start*dt)"
    T_start::FT1
    "end time to gather statistics (T_end = N_end*dt)"
    T_end::FT2
end

#########################################################################
############################ Model Functions ############################
#########################################################################

# Forward pass of forward model
# Inputs: 
# - params: structure with F 
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_forward(
    params::EnsembleMemberConfig,
    x0::VorM,
    config::LorenzConfig,
    observation_config::ObservationConfig,
) where {VorM <: AbstractVecOrMat}
    # run the Lorenz simulation
    xn = lorenz_solve(params, x0, config)
    # Get statistics
    gt = stats(xn, config, observation_config)
    return gt
end

#Calculates statistics for forward model output
# Inputs: 
# - xn: timeseries of states for length of simulation through Lorenz96
function stats(xn::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
    T_start = observation_config.T_start
    T_end = observation_config.T_end
    dt = config.dt
    N_start = Int(ceil(T_start / dt))
    N_end = Int(ceil(T_end / dt))
    xn_stat = xn[:, N_start:N_end]
    N_state = size(xn_stat, 1)
    gt = zeros(2 * N_state)
    gt[1:N_state] = mean(xn_stat, dims = 2)
    gt[(N_state + 1):(2 * N_state)] = std(xn_stat, dims = 2)
    return gt
end

# Forward pass of the Lorenz 96 model
# Inputs: 
# - params: structure with F 
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # Initialize    
    nstep = Int(ceil(config.T / config.dt))
    state_dim = isa(x0, AbstractVector) ? length(x0) : size(x0, 1)
    xn = zeros(size(x0, 1), nstep + 1)
    xn[:, 1] = x0

    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(params, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with F 
# - x: current state
function f(params::EnsembleMemberConfig, x::VorM) where {VorM <: AbstractVecOrMat}
    N = length(x)
    f = zeros(N)
    # Loop over N positions
    for i in 3:(N - 1)
        f[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + forcing(params, x, i)  
    end
    # Periodic boundary conditions
    f[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] +  forcing(params, x, 1) 
    f[2] = -x[N] * x[1] + x[1] * x[3] - x[2] +  forcing(params, x, 2) 
    f[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] +  forcing(params, x, N) 
    # Output
    return f
end

# RK4 solve
# Inputs: 
# - params: structure with F 
# - xold: current state
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function RK4(params::EnsembleMemberConfig, xold::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
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