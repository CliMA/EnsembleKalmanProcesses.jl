# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics
using Flux
using BSON

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include("Lorenz96.jl") # Contains Lorenz 96 source code

########################################################################
######################### Choose problem type ##########################
########################################################################

## Define forcing:
cases = [
    "const-force",
    "vec-force",
    "flux-force",
]
case = cases[1]

if case == "const-force"
    nx = 40  #dimensions of parameter vector
    phi = ConstantEMC(8.0) # forcing
    phi_structure = nothing
    prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
    T = 14.0
elseif case == "vec-force"
    nx = 40  # dimensions of parameter vector
    sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx) 
    phi = VectorEMC(sinusoid)
    phi_structure = nothing
    pl, psig = 2.0, 3.0
    B = zeros(nx, nx) # prior covariance
    for ii in 1:nx
        for jj in 1:nx
            B[ii, jj] = psig^2 * exp(-abs(ii - jj) / pl)
        end
    end
    mu = 8.0 * ones(nx) # prior mean
    #Creating prior distribution
    distribution = Parameterized(MvNormal(mu, B))
    constraint = repeat([no_constraint()], 40)
    name = "ml96_prior"
    prior = ParameterDistribution(distribution, constraint, name)
    T = 54.0
# elseif case == "flux-force"
#     nx = 100  #dimensions of parameter vector
#     sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
#     from_file=true
#     # from_file
#     if from_file
#         filename = "filename"
#         phi_structure = BSON.@load "$(filename).bson" model
#         prior_mean, prior_cov = BSON.@load "$(filename).bson" prior_mean, prior_cov
#         prior = ParameterDistribution(..)
#     else
#         input_dim = 1
#         phi_structure = Chain(
#             Dense(input_dim => 20, tanh),                 
#             Dense(20 => 1),                       
#         )
#         prior = constrained_gaussian("params", 0,1,-Inf, Inf, repeat=input_dim)
#     end
#     T = 504.0
#     phi = FluxEMC(model) # need to call build_forcing(parameters, model)
end


########################################################################
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

#Creating my sythetic data

t = 0.01  #time step
T_long = 1000.0  #total time 
picking_initial_condition = LorenzConfig(t, T_long) 
x_initial = rand(rng_i, Normal(0.0, 1.0), nx) # initial condition for spinning up Lorenz system
x_spun_up = lorenz_solve(phi, x_initial, picking_initial_condition) # spinning up Lorenz system

x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data

ny = nx * 2   #number of data points
lorenz_config_settings = LorenzConfig(t, T)

# construct how we compute Observations
T_start = 4.0  #2*max
T_end = T
observation_config = ObservationConfig(T_start, T_end)
y = lorenz_forward(phi, x0, lorenz_config_settings, observation_config) # synthetic data

#Observation covariance R
model_out_vars = (0.1 * y) .^ 2
R = Diagonal(model_out_vars)
R_sqrt = sqrt(R)
R_inv_var = sqrt(inv(R))


# Need a way to perturb the initial condition when doing the EKI updates
# Solving for initial condition perturbation covariance
covT = 2000.0  #time to simulate to calculate a covariance matrix of the system
cov_solve = lorenz_solve(phi, x0, LorenzConfig(t, covT))
ic_cov = 0.1 * cov(cov_solve, dims = 2)
ic_cov_sqrt = sqrt(ic_cov)