using Distributions
using JLD
using ArgParse
using NCDatasets
using LinearAlgebra
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

include(joinpath(@__DIR__, "helper_funcs.jl"))

"""
ek_update(iteration_::Int64)

Update CLIMAParameters ensemble using Ensemble Kalman Inversion,
return the ensemble and the parameter names.
"""
function ek_update(iteration_::Int64)
  
    # Recover ensemble from last iteration, [N_ens, N_params]
    u_mean, uu_cov, _, _ = read_params( iteration_-1)
    

    # Get outputs
    g_ens = zeros(N_ens, N_y)
    for ens_index = 1:N_ens
        g_ens[ens_index, :] = read_solver_output(iteration_-1,  ens_index)
    end
    g_ens = Array(g_ens')


    # Construct EKP
    process = Unscented(u_mean, uu_cov; prior_mean = prior_mean, α_reg = α_reg, update_freq = update_freq, sigma_points = sigma_points_type)

    obs_mean, obs_noise_cov = read_observation()

    ukiobj = EnsembleKalmanProcess(obs_mean, obs_noise_cov, process)
    
    
    # Advance EKP
    update_ensemble!(ukiobj, g_ens)

    
    # Save mean, covariance, and sigma particles after the analysis step
    save_params(ukiobj, iteration_)
    
    # Get new step
    u_p_ens_new = get_u_final(ukiobj)
    constraint_u_p_ens_new = constraint(u_p_ens_new)
    write_solver_input_files(constraint_u_p_ens_new)
    
    return u_p_ens_new
end



# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--iteration"
    help = "Calibration iteration number"
    arg_type = Int
    default = 1
end
parsed_args = parse_args(ARGS, s)
iteration_ = parsed_args["iteration"]



ek_update(iteration_)
