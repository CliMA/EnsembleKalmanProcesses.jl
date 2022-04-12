#Template for a perfect model example:


# Import external modules

# Import CES modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
using EnsembleKalmanProcesses.DataStorage

# include model or data
include("GModel.jl")

##
## Create your perfect model "truth"
##

true_params = 0.0

GModel.set_properties(initial_conditions, boundary_conditions, fixed_params)

# Obtain samples of truth
truth_data = GModel.run_G_ensemble(true_params, GModel_settings, ensemble_size = 1)

# Calculate the observational noise on the data.
noise_covariance = prescribed_observational_noise_covariance

##
## Store the observation and covariance
##

truth = Observations.Obs(truth_data, observational_noise_covariance, data_names)
truth_sample = truth.samples[1]

##
## Now define the priors:
##
prior_dict = Dict("distribution" => Parameterized(Normal(0, 1)), "constraint" => bounded(1, 2), "name" => "")
priors = ParameterDistribution(prior_dict)

##
## Perform Ensemble_Kalman_Inversion
##

# initial ensemble of parameters
params = construct_initial_ensemble(priors, N_ens; rng_seed = rng_seed)

# constructor for the EKI object
ekiobj =
    EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_params, truth_sample, truth.obs_noise_cov, Inversion())

#outer EKI loop
for i in 1:N_iterations
    # theta^n_i -> G(theta^n_i)
    physical_params_i = transform_unconstrained_to_constrained(priors, params)
    g_ens = GModel.run_G_ensemble(physical_params_i, lorenz_settings_G)

    # G(theta^n_i) -> theta^{n+1}_i
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)

    err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)

end

##
## store results
##

u_stored = get_u(ekiobj, return_array = false)
g_stored = get_g(ekiobj, return_array = false)

#solution:
final_params = get_u_final(ekiobj)
solution = mean(final_params, dims = 2)

physical_final_params = transform_unconstrained_to_constrained(priors, final_ensemble)
physical_solution = mean(physical_final_soluation)
