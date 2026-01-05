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
case = cases[2]

if case == "const-force"
    nx = 40  #dimensions of parameter vector
    nu = 1
    phi = ConstantEMC(8.0) # forcing
    phi_structure = nothing
    prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
    T = 14.0
    inff = 2
elseif case == "vec-force"
    nx = 40  # dimensions of parameter vector
    nu = nx
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
    inff = 2
# elseif case == "flux-force"
#     nx = 100  #dimensions of parameter vector
#     nu = 61
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
#     inff = 2.5
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
window = T_end - T_start
T_R = 10*window*ny + T_start
R_config = LorenzConfig(t, T_R)
R_run = lorenz_solve(phi, x_initial, R_config)
R_sample_size = Int(ceil(10*ny))
R_samples = zeros(ny, R_sample_size)
for ii in 1:R_sample_size
    local_obs_config = ObservationConfig(T_start + (ii -1)*window, T_start + ii*window)
    R_samples[:,ii] = stats(R_run, R_config, local_obs_config)
end
R = cov(R_samples, dims = 2)*inff
R_sqrt = sqrt(R)
R_inv_var = sqrt(inv(R))


# Need a way to perturb the initial condition when doing the EKI updates
# Solving for initial condition perturbation covariance
covT = 2000.0  #time to simulate to calculate a covariance matrix of the system
cov_solve = lorenz_solve(phi, x0, LorenzConfig(t, covT))
ic_cov = 0.1 * cov(cov_solve, dims = 2)
ic_cov_sqrt = sqrt(ic_cov)

########################################################################
########################### Running EKI Race ###########################
########################################################################

# EKP parameters (move towards top for having user choice, put info statement at top)
N_ens_sizes = [100] #, 55] # number of ensemble members (should be problem dependent)
N_iter = 20 # number of EKI iterations
tolerance = 1.0

rng_seeds = [3] #, 15] #, 42, 101]

conv_alg_iters = zeros(4, length(N_ens_sizes), length(rng_seeds)) #count how many iterations it takes to converge (per algorithm, per rand seed, per ense size)
final_parameters = zeros(4, length(N_ens_sizes), length(rng_seeds), nu)
final_model_output = zeros(4, length(N_ens_sizes), length(rng_seeds), ny)

for (rr, rng_seed) in enumerate(rng_seeds)
    @info "Random seed: $(rng_seed)"
    rng = MersenneTwister(rng_seed)

    for (ee, N_ens) in enumerate(N_ens_sizes)
        # initial parameters: N_params x N_ens
        initial_params = construct_initial_ensemble(rng, prior, N_ens)

        # prior_mean = isa(mean(prior), AbstractVector) ? mean(prior) : [mean(prior)] # mean of unconstrained distribution
        # prior_cov = Matrix(cov(prior)) # cov of unconstrained distribution

        methods =
            [Inversion(prior), TransformInversion(prior), GaussNewtonInversion(prior), Unscented(prior; impose_prior = true)] # GaussNewtonInversion(prior)

        @info "Ensemble size: $(N_ens)"
        for (kk, method) in enumerate(methods)
            if isa(method, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(
                    y,
                    R,
                    method;
                    rng = copy(rng),
                    verbose = true,
                    accelerator = DefaultAccelerator(),
                    localization_method = NoLocalization(),
                    scheduler = DefaultScheduler(),
                )
            else
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_params,
                    y,
                    R,
                    method;
                    rng = copy(rng),
                    verbose = true,
                    accelerator = DefaultAccelerator(),
                    localization_method = NoLocalization(),
                    scheduler = DefaultScheduler(),
                )
            end
            Ne = get_N_ens(ekpobj)

            count = 0
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)

                # Calculating RMSE_e
                ens_mean = mean(params_i, dims = 2)[:]
                # forcing = build_forcing(ens_mean, phi_structure)
                forcing = build_forcing(phi, ens_mean, phi_structure)
                G_ens_mean = lorenz_forward(
                    forcing,
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
                    lorenz_config_settings,
                    observation_config,
                )
                RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(size(y, 1))
                @info "RMSE (at G(u_mean)): $(RMSE_e)"
                # Convergence criteria
                if RMSE_e < tolerance
                    conv_alg_iters[kk, ee, rr] = count * Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    final_model_output[kk, ee, rr, :] = G_ens_mean
                    break
                end

                # If RMSE convergence criteria is not satisfied 
                G_ens = hcat(
                    [
                        lorenz_forward(
                            build_forcing(phi, params_i[:, j], phi_structure),
                            (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j],
                            lorenz_config_settings,
                            observation_config,
                        ) for j in 1:Ne
                    ]...,
                )
                # Update 
                EKP.update_ensemble!(ekpobj, G_ens)
                count = count + 1

                # Calculate RMSE_f 
                # RMSE_f = get_error_metrics(ekpobj)["avg_rmse"][end]
                RMSE_f = norm(R_inv_var * (y - mean(G_ens, dims = 2))) / sqrt(size(y, 1))
                @info "RMSE (at mean(G(u)): $(RMSE_f)"
                # Convergence criteria
                if RMSE_f < tolerance
                    conv_alg_iters[kk, ee, rr] = count * Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    final_model_output[kk, ee, rr, :] = G_ens_mean
                    break
                end
            end

            final_ensemble = get_ϕ_final(prior, ekpobj)
        end
    end
end