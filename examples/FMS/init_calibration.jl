using Distributions
using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
include(joinpath(@__DIR__, "helper_funcs.jl"))





process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq, sigma_points = "symmetric")
ukiobj = EKP.EnsembleKalmanProcess(u_init, obs_mean, obs_noise_cov, process)


save_param(ukiobj, 0)


u_p_ens_new = get_u_final(ukiobj)


# transform parameters  

constraint_u_p_ens_new = constraint(u_p_ens_new)

# update input files

write_fms_input_files(constraint_u_p_ens_new)
