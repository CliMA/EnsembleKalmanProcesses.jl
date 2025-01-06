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
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    "state-dependent-forcing"
    F::VV
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
# - params: structure with F (state-dependent-forcing vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_forward(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
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
    gt = zeros(2*N_state)
    gt[1:N_state] = mean(xn_stat, dims=2) 
    gt[N_state+1:2*N_state]= std(xn_stat, dims=2)
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
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init) 

#Creating my sythetic data
#initalize model variables
nx = 40  #dimensions of parameter vector
gamma = 8 .+ 6*sin.((4*pi*range(0, stop=nx - 1, step=1))/nx)  #forcing (Needs to be of type EnsembleMemberConfig)
true_parameters = EnsembleMemberConfig(gamma)

t = 0.01  #time step
T_long = 1000.0  #total time 
picking_initial_condition = LorenzConfig(t, T_long)

#beginning state
x_initial = rand(rng_i, Normal(0.0, 1.0), nx)

#Find the initial condition for my data
x_spun_up = lorenz_solve(true_parameters, x_initial, picking_initial_condition)  #Need to make LorenzConfig object with t, T_long

#intital condition used for the data
x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data
#Creating my sythetic data
T = 504.0
ny = nx*2   #number of data points
lorenz_config_settings = LorenzConfig(t, T)

# construct how we compute Observations
T_start = 4.0  #2*max
T_end = T
observation_config = ObservationConfig(T_start,T_end)


model_out_y = lorenz_forward(true_parameters, x0, lorenz_config_settings, observation_config) 

#Observation covariance R
model_out_vars = (0.1*model_out_y).^2
R = Diagonal(model_out_vars)
R_sqrt = sqrt(R)
R_inv_var = sqrt(inv(R))

#Observations y
y = model_out_y  + R_sqrt*rand(rng_i, Normal(0.0, 1.0), ny)

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
# Solving for initial condition perturbation covariance
covT = 2000.0  #time to simulate to calculate a covariance matrix of the system
cov_solve = lorenz_solve(true_parameters, x0, LorenzConfig(t, covT))
ic_cov = 0.1*cov(cov_solve, dims = 2)
ic_cov_sqrt = sqrt(ic_cov)

########################################################################
############################# Running GNKI #############################
########################################################################

# EKP parameters
N_ens_sizes = [50, 90] #, 110, 130, 150] # number of ensemble members
N_iter = 20 # number of EKI iterations
tolerance = 1.0

methods = [Inversion(), TransformInversion(), GaussNewtonInversion(prior), 
            Unscented(prior, impose_prior = true)]

rng_seeds = [2, 15] #, 42, 101]

conv_alg_iters = zeros(4, length(N_ens_sizes), length(rng_seeds)) #count how many iterations it takes to converge (per algorithm, per rand seed, per ense size)
final_parameters = zeros(4, length(N_ens_sizes), length(rng_seeds), nx)

for (rr, rng_seed) in enumerate(rng_seeds)
    @info "Random seed: $(rng_seed)"
    rng = MersenneTwister(rng_seed)

    for (ee, N_ens) in enumerate(N_ens_sizes)
        # initial parameters: N_params x N_ens
        initial_params = construct_initial_ensemble(rng, prior, N_ens)

        @info "Ensemble size: $(N_ens)"
        for (kk, method) in enumerate(methods)
            if isa(method, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(y, R, method; 
                            rng = copy(rng), verbose = true, accelerator = DefaultAccelerator(), 
                            localization_method = NoLocalization(), scheduler = DefaultScheduler())
            else    
                ekpobj = EKP.EnsembleKalmanProcess(initial_params, y, R, method; 
                            rng = copy(rng), verbose = true, accelerator = DefaultAccelerator(), 
                            localization_method = NoLocalization(), scheduler = DefaultScheduler())
            end
            Ne = get_N_ens(ekpobj)

            count = 0
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)

                # Calculating RMSE_e
                ens_mean = mean(params_i, dims = 2)[:]
                G_ens_mean = lorenz_forward(EnsembleMemberConfig(ens_mean), 
                            x0 .+ ic_cov_sqrt*rand(rng, Normal(0.0, 1.0), nx, 1), 
                            lorenz_config_settings, observation_config)
                RMSE_e = norm(R_inv_var*(y - G_ens_mean[:]))/sqrt(size(y, 1))
                @info "RMSE (at G(u_mean)): $(RMSE_e)"
                # Convergence criteria
                if RMSE_e < tolerance
                    conv_alg_iters[kk, ee, rr] = count*Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    break
                end
                
                # If RMSE convergence criteria is not satisfied 
                G_ens = hcat([lorenz_forward(EnsembleMemberConfig(params_i[:, j]), 
                                (x0 .+ ic_cov_sqrt*rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j], 
                                lorenz_config_settings, observation_config) for j in 1:Ne]...)
                # Update 
                EKP.update_ensemble!(ekpobj, G_ens)
                count = count + 1

                # Calculate RMSE_f
                RMSE_f = sqrt(get_error(ekpobj)[end]/size(y, 1))
                @info "RMSE (at mean(G(u)): $(RMSE_f)"
                # Convergence criteria
                if RMSE_f < tolerance 
                    conv_alg_iters[kk, ee, rr] = count*Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    break
                end
            end

            final_ensemble = get_ϕ_final(prior, ekpobj)
        end
    end
end

# Create a plot
plot(range(0, nx -1, step = 1), 
    [gamma final_parameters[1,1,1,:] final_parameters[2,1,1,:] final_parameters[3,1,1,:] final_parameters[4,1,1,:]], 
    label = ["solution" "TEKI" "ETKI" "GNKI" "UKI"], 
    color = [:black :green :blue :orange :purple], 
    xlabel = "Spatial index", 
    ylabel= "Gamma",
    title = "Optimized parameters for ensemble size = 50",
    show = true)

readline()

plot(range(0, nx -1, step = 1), 
    [gamma final_parameters[1,1,2,:] final_parameters[2,1,2,:] final_parameters[3,1,2,:] final_parameters[4,1,2,:]], 
    label = ["solution" "TEKI" "ETKI" "GNKI" "UKI"], 
    color = [:black :green :blue :orange :purple], 
    xlabel = "Spatial index", 
    ylabel= "Gamma",
    title = "Optimized parameters for ensemble size = 90",
    show = true,
    reuse = false)

readline()

plot(N_ens_sizes, 
    [mean(conv_alg_iters[1, :, :], dims = 2) mean(conv_alg_iters[2, :, :], dims = 2) mean(conv_alg_iters[3, :, :], dims = 2)],
    label = ["TEKI" "ETKI" "GNKI"], 
    color = [:green :blue :orange], 
    xlabel = "Ensemble size", 
    ylabel= "Number of forward runs",
    title = "EKI Race",
    show = true,
    reuse = false)
plot!([81], [mean(conv_alg_iters[4, :, :])],
    label = "UKI",
    color = :purple,)

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

































