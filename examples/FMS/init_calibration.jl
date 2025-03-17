using Distributions
using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
include(joinpath(@__DIR__, "helper_funcs.jl"))





process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq, sigma_points = sigma_points_type)

#u_init = EnsembleKalmanProcesses.construct_sigma_ensemble(process, prior_mean, prior_cov)

obs_mean, obs_noise_cov = read_observation()

ukiobj = EnsembleKalmanProcess(obs_mean, obs_noise_cov, process)

save_params(ukiobj, 0)


u_p_ens_new = get_u_final(ukiobj)


# transform parameters  

constraint_u_p_ens_new = constraint(u_p_ens_new)

# update input files

write_solver_input_files(constraint_u_p_ens_new)
