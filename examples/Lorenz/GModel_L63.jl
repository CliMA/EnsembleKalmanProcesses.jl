using Random
using Distributions
using LinearAlgebra
using Statistics

# G(θ) = H(Ψ(θ,x₀,t₀,t₁))
# y = G(θ) + η
#
# The Lorenz 1963 system:
#   dx/dt = σ(y - x)
#   dy/dt = x(ρ - z) - y
#   dz/dt = xy - βz
# The unknown parameters are ρ (rho) and β (beta); σ (sigma) is held at its classical value.

# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{FT <: Real}
    "Prandtl number"
    sigma::FT
    "Rayleigh number"
    rho::FT
    "Geometric factor"
    beta::FT
end

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
# - params: structure with sigma, rho, beta
# - x0: initial condition vector
# - config: structure including dt (timestep) and T (total time)
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

# Calculates statistics for forward model output
# Inputs:
# - xn: timeseries of states for length of simulation through Lorenz63
function stats(xn::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
    T_start = observation_config.T_start
    T_end = observation_config.T_end
    dt = config.dt
    N_start = Int(ceil(T_start / dt))
    N_end = Int(ceil(T_end / dt))
    xn_stat = xn[:, N_start:N_end]
    gt = zeros(eltype(xn_stat), 9) # 3 means, 3 variances, 3 covariances
    gt[1:3] = mean(xn_stat, dims = 2)
    xn_stat_cov = cov(xn_stat, dims = 2)
    gt[4:6] = diag(xn_stat_cov)
    gt[7:8] = xn_stat_cov[1, 2:3]
    gt[9] = xn_stat_cov[2, 3]
    return gt
end

# Forward pass of the Lorenz 63 model
# Inputs:
# - params: structure with sigma, rho, beta
# - x0: initial condition vector
# - config: structure including dt (timestep) and T (total time)
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # Initialize
    nstep = Int(ceil(config.T / config.dt))
    xn = zeros(promote_type(typeof(params.rho), eltype(x0)), length(x0), nstep + 1)
    xn[:, 1] = x0
    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(params, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 63 system
# f = dx/dt
# Inputs:
# - params: structure with sigma, rho, beta
# - x: current state
function f(params::EnsembleMemberConfig, x::VorM) where {VorM <: AbstractVecOrMat}
    out = zeros(promote_type(typeof(params.rho), eltype(x)), 3)
    out[1] = params.sigma * (x[2] - x[1])
    out[2] = x[1] * (params.rho - x[3]) - x[2]
    out[3] = x[1] * x[2] - params.beta * x[3]
    return out
end

# RK4 solve
function RK4(params::EnsembleMemberConfig, xold::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
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
