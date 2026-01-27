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

include("Lorenz63.jl") # Contains Lorenz 96 source code

########################################################################
############### Choose problem type and structure ######################
########################################################################

N_ens_sizes = [20] # list of number of ensemble members (should be problem dependent)
N_iter = 20 # maximum number of EKI iterations allowed
target_rmse = 1.0 # target RMSE 
rng_seeds = [3] # list of random seeds
@info "Running Lorenz 63 problem"
@info "Maximum number of EKI iterations: $N_iter"
@info "RMSE target: $target_rmse"
configuration =
    Dict("N_iter" => N_iter, "N_ens_sizes" => N_ens_sizes, "target_rmse" => target_rmse, "rng_seeds" => rng_seeds)

nx = 3  # dimensions of parameter vector
nu = 2
u = EnsembleMemberConfig([28.0, 8.0 / 3.0])

prior_mean = [3.3, 1.2]
prior_cov = [
    0.15^2 0
    0 0.5^2
]
#Creating prior distribution
distribution = Parameterized(MvNormal(prior_mean, prior_cov))
constraint = repeat([no_constraint()], 2)
name = "l63_prior"
prior = ParameterDistribution(distribution, constraint, name)
T = 40.0



########################################################################
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

#Creating my sythetic data

t = 0.01  #time step
T_long = 1000.0  #total time 
picking_initial_condition = LorenzConfig(t, T_long)
x_initial = rand(rng_i, Normal(0.0, 1.0), nx) # initial condition for spinning up Lorenz system
x_spun_up = lorenz_solve(u, x_initial, picking_initial_condition) # spinning up Lorenz system

x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data

ny = 9   #number of data points
lorenz_config_settings = LorenzConfig(t, T)

# construct how we compute Observations
T_start = 30.0
T_end = T
observation_config = ObservationConfig(T_start, T_end)
y = lorenz_forward(u, x0, lorenz_config_settings, observation_config) # synthetic data

#Observation covariance R
multiple = 36
window = T_end - T_start
T_R = multiple * window + T_start
R_config = LorenzConfig(t, T_R)
R_run = lorenz_solve(u, x_initial, R_config)
R_sample_size = Int(ceil(multiple))
R_samples = zeros(ny, R_sample_size)
for ii in 1:R_sample_size
    local_obs_config = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
    R_samples[:, ii] = stats(R_run, R_config, local_obs_config)
end
R = cov(R_samples, dims = 2)
R_sqrt = sqrt(R)
R_inv_var = sqrt(inv(R))


# Need a way to perturb the initial condition when doing the EKI updates
# Solving for initial condition perturbation covariance
covT = 2000.0  #time to simulate to calculate a covariance matrix of the system
cov_solve = lorenz_solve(u, x0, LorenzConfig(t, covT))
ic_cov = 0.1 * cov(cov_solve, dims = 2)
ic_cov_sqrt = sqrt(ic_cov)

########################################################################
########################### Running EKI Race ###########################
########################################################################

# Counters
conv_alg_iters = zeros(4, length(N_ens_sizes), length(rng_seeds)) #count how many iterations it takes to converge (per algorithm, per rand seed, per ense size)
final_parameters = zeros(4, length(N_ens_sizes), length(rng_seeds), nu)
final_model_output = zeros(4, length(N_ens_sizes), length(rng_seeds), ny)

method_names = [
    "Inversion(prior)",
    "TransformInversion(prior)",
    "GaussNewtonInversion(prior)",
    "Unscented(prior; impose_prior=true)",
]


for (rr, rng_seed) in enumerate(rng_seeds)
    @info "Random seed: $(rng_seed)"
    rng = MersenneTwister(rng_seed)

    for (ee, N_ens) in enumerate(N_ens_sizes)
        # initial parameters: N_params x N_ens
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        methods = [
            Inversion(prior),
            TransformInversion(prior),
            GaussNewtonInversion(prior),
            Unscented(prior; impose_prior = true),
        ]

        @info "Ensemble size: $(N_ens)"
        for (kk, method) in enumerate(methods)
            if isa(method, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(
                    y,
                    R,
                    deepcopy(method);
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
                    deepcopy(method);
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
                G_ens_mean = lorenz_forward(
                    EnsembleMemberConfig(exp.(ens_mean)),
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
                    lorenz_config_settings,
                    observation_config,
                )
                RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(size(y, 1))
                @info "RMSE (at G(u_mean)): $(RMSE_e)"
                # Convergence criteria
                if RMSE_e < target_rmse
                    conv_alg_iters[kk, ee, rr] = count * Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    final_model_output[kk, ee, rr, :] = G_ens_mean
                    break
                end

                # If RMSE convergence criteria is not satisfied 
                G_ens = hcat(
                    [
                        lorenz_forward(
                            EnsembleMemberConfig(exp.(params_i[:, j])),
                            (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j],
                            lorenz_config_settings,
                            observation_config,
                        ) for j in 1:Ne
                    ]...,
                )
                # Update 
                EKP.update_ensemble!(ekpobj, G_ens)
                count = count + 1

                # # Calculate RMSE_f to the mean G(u) if desired
                # RMSE_f = sqrt(get_error_metrics(ekpobj)["loss"][end]) # equivalently
                # @info "RMSE (at mean(G(u)): $(RMSE_f)"
                # # Convergence criteria
                # if RMSE_f < target_rmse
                #     conv_alg_iters[kk, ee, rr] = count * Ne
                #     final_parameters[kk, ee, rr, :] = ens_mean
                #     final_model_output[kk, ee, rr, :] = G_ens_mean
                #     break
                # end
            end

            final_ensemble = get_ϕ_final(prior, ekpobj)
        end
    end
end

# Saving data:
using Dates
date_of_exp = today()
data_filename = joinpath(output_dir, "l63_output_$(case)_$(today()).jld2")
JLD2.save(
    data_filename,
    "configuration",
    configuration,
    "method_names",
    method_names,
    "conv_alg_iters",
    conv_alg_iters,
    "final_parameters",
    final_parameters,
    "final_model_output",
    final_model_output,
)
