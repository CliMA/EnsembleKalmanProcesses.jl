# Import modules
using LinearAlgebra
using Statistics
using Random, Distributions
using Flux



# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
abstract type EnsembleMemberConfig end
# struct EnsembleMemberConfig{FT}
#    val::FT
# end

# Sub-type of ensemble config for constant forcing
struct ConstantEMC{FT <: Real} <: EnsembleMemberConfig
    val::FT
end
build_forcing(::T, val::FT, args...) where {T <: ConstantEMC, FT <: Real} = ConstantEMC(val)
build_forcing(::T, val::FT, args...) where {T <: ConstantEMC, FT <: AbstractVector} = ConstantEMC(val[1])

# Sub-type of ensemble config for spatially-dependent forcing
struct VectorEMC{VV <: AbstractVector} <: EnsembleMemberConfig
    val::VV
end
build_forcing(::T, val::VV, args...) where {T <: VectorEMC, VV <: AbstractVector} = VectorEMC(val)

# Sub-type of ensemble config for spatially-dependent forcing with neural network approximation
struct FluxEMC{FC <: Flux.Chain, VV <: AbstractVector} <: EnsembleMemberConfig
    model::FC
    sample_range::VV
end
function build_forcing(::T, params, model, sample_range) where {T <: FluxEMC}
    _, reconstructor = Flux.destructure(model)
    return FluxEMC(reconstructor(params), Float32.(sample_range))
end

# Constant-global
forcing(params::ConstantEMC, x, i) = params.val
forcing(params::ConstantEMC, x) = repeat([params.val], length(x))

# Constant-vector
forcing(params::VectorEMC, x, i) = params.val[i]
forcing(params::VectorEMC, x) = params.val

# Flux
forcing(params::FluxEMC, x, i) = Float64(params.model([params.sample_range[i]])[1])
forcing(params::FluxEMC, x) = Float64.(params.model([sr])[1] for sr in params.sample_range)




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
    forcing_vec = forcing(params, x0) # not state dependent so evaluate once here
    for j in 1:nstep
        xn[:, j + 1] = RK4(forcing_vec, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with F 
# - x: current state
function f(forcing_vec::VV, x::VorM) where {VV <: AbstractVector, VorM <: AbstractVecOrMat}
    N = length(x)
    f = zeros(N)
    # Loop over N positions
    for i in 3:(N - 1)
        f[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + forcing_vec[i]
    end
    # Periodic boundary conditions
    f[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] + forcing_vec[1]
    f[2] = -x[N] * x[1] + x[1] * x[3] - x[2] + forcing_vec[2]
    f[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] + forcing_vec[N]

    # Output
    return f
end

# RK4 solve
# Inputs: 
# - params: structure with F 
# - xold: current state
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function RK4(forcing_vec::VV, xold::VorM, config::LorenzConfig) where {VV <: AbstractVector, VorM <: AbstractVecOrMat}
    N = length(xold)
    dt = config.dt

    # Predictor steps (note no time-dependence is needed here)
    k1 = f(forcing_vec, xold)
    k2 = f(forcing_vec, xold + k1 * dt / 2.0)
    k3 = f(forcing_vec, xold + k2 * dt / 2.0)
    k4 = f(forcing_vec, xold + k3 * dt)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end


# Neural network functions 
function train_network(model, x_train, y_train)
    loss(model, x, y) = Flux.Losses.mse(model(x), y)

    # Reshape x_train and y_train for Flux compatibility
    x_train = reshape(x_train, 1, :)
    y_train = reshape(y_train, 1, :)
    x_train = Float32.(x_train)
    y_train = Float32.(y_train)

    opt = Flux.setup(Adam(), model)
    data = Flux.DataLoader((x_train, y_train), batchsize = 32, shuffle = true)  # train the model

    # Train the model over multiple epochs
    epochs = 5000
    for epoch in 1:epochs
        Flux.train!(loss, model, data, opt)
    end

    params, _ = Flux.destructure(model)
    return model, params
end
