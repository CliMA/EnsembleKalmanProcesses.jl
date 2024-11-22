include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

# G(θ) = H(Ψ(θ,x₀,t₀,t₁))
# y = G(θ) + η

# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    dt::FT1
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    "state-dependent-forcing"
    F::VV
end

# This will change for different "Observations" of Lorenz
struct ObservationConfig{FT1 <: Real, FT2 <: Real} 
    t_start::FT1
    t_end::FT2
end
#########################################################################
############################ Model Functions ############################
#########################################################################

# Forward pass of forward model
# Inputs: 
# - params: structure with F (state-dependent-forcing vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_forward(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # run the Lorenz simulation
    xn = lorenz_solve(params, x0, config)
    # Get statistics
    gt = stats(xn)
    return gt
end

#Calculates statistics for forward model output
# Inputs: 
# - xn: timeseries of states for length of simulation through Lorenz96
function stats(xn::VorM) where {VorM <: AbstractVecOrMat}
    N = size(xn, 1)
    gt = zeros(2*N)
    gt[1:N] = mean(xn, dims=2) 
    gt[N+1:2*N]= std(xn, dims=2)
    return gt
end

# Forward pass of the Lorenz 96 model
# Inputs: 
# - params: structure with F (state-dependent-forcing vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
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
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
# - x: current state
function f(params::EnsembleMemberConfig, x::VorM) where {VorM <: AbstractVecOrMat}
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
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
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

########################################################################
########################## Ensemble Functions ##########################
########################################################################
# Running ensemble members through the forward function
# Inputs: 
# - params: array of structures with F (state-dependent-forcing vector) 
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
# - nd: size of model output (integer)
# - N_ens: number of ensemble members (integer)
function run_ensembles(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig, nd, N_ens) where {VorM <: AbstractVecOrMat}
    g_ens = zeros(nd, N_ens)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[:, i] = lorenz_forward(params[:, i], x0, config)
    end
    return g_ens
end
# Note: I'm not sure how to make an array of EnsembleMemberConfig objects?

########################################################################
############################ Problem setup #############################
########################################################################
rng_seed = 3
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

#Creating my sythetic data
#initalize model variables
nx = 40  #dimensions of parameter vector
gamma = 8 .+ 6*sin.((4*pi*range(0, stop=nx, step=1))/nx)  #forcing (Needs to be of type EnsembleMemberConfig)
true_parameters = EnsembleMemberConfig(gamma)

t = 0.01  #time step
T_long = 1000.0  #total time 
picking_initial_condition = LorenzConfig(t, T_long)

#beginning state
int_state = rand(Normal(0.0, 1.0), nx)

#Find the initial condition for my data
spin_up_array = lorenz_solve(true_parameters, int_state, picking_initial_condition)  #Need to make LorenzConfig object with t, T_long

#intital condition used for the data
x0 = spin_up_array[:, end]  #last element of the run is the initial condition for creating the data
#Creating my sythetic data
T = 500.0
ny = nx*2   #number of data points
lorenz_config_settings = LorenzConfig(t, T)

model_out_y = lorenz_forward(true_parameters, x0, lorenz_config_settings) 

#Observation covariance R
model_out_vars = (0.1*model_out_y).^2
R = Diagonal(model_out_vars)
R_sqrt = sqrt(R)

#Observations y
y = model_out_y  + R_sqrt*rand(Normal(0.0, 1.0), ny)

pl = 2.0
psig = 3.0
#Prior covariance
B = zeros(nx,nx)
for ii in 1:nx 
    for jj in 1:nx
        B[ii, jj] = psig^2 * exp(-abs(ii - jj)/pl)  
    end 
end
B_sqrt = sqrt(B)

#Prior mean
mu = 8.0*ones(nx) 

#Creating prior distribution
distribution = Parameterized(MvNormal(mu, B))
constraint = repeat([no_constraint()], 40)
name = "ml96_prior"

prior = ParameterDistribution(distribution, constraint, name)

# Need a way to perturb the initial condition when doing the EKI updates

########################################################################
############################# Running GNKI #############################
########################################################################

# EKP parameters
N_ens = 50# number of ensemble members
N_iter = 4 # number of EKI iterations

# initial parameters: N_params x N_ens
initial_params = construct_initial_ensemble(rng, prior, N_ens)

ekiobj = EKP.EnsembleKalmanProcess(initial_params, y, R, GaussNewtonInversion(prior); 
            rng = rng, verbose = true, accelerator = DefaultAccelerator(), 
            localization_method = NoLocalization())

for i in 1:N_iter
    params_i = get_ϕ_final(prior, ekiobj)

    G_ens = hcat([lorenz_forward(EnsembleMemberConfig(params_i[:, j]), 
                    x0, lorenz_config_settings) for j in 1:N_ens]...)

    EKP.update_ensemble!(ekiobj, G_ens)
end

final_ensemble = get_ϕ_final(prior, ekiobj)

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

































