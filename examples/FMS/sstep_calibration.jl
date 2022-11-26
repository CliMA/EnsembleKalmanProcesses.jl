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
    u_ens = read_params( iteration_)
    u_ens = Array(u_ens')


    # Get outputs
    g_ens = zeros(N_ens, N_y)
    for ens_index = 1:N_ens
        g_ens[ens_index, :] = read_fms_output( ens_index)
    end
    g_ens = Array(g_ens')


    # Construct EKP
    process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq, sigma_points = "symmetric")
    ukiobj = EKP.EnsembleKalmanProcess(u_ens, obs_mean, obs_noise_cov, process)
    
    
    # Advance EKP
    update_ensemble!(ukiobj, g_ens)

    
    # Save mean, covariance, and sigma particles after the analysis step
    save_param(ukiobj, iteration_)
    
    # Get new step
    u_p_ens_new = get_u_final(ukobj)
    constraint_u_p_ens_new = constraint(u_p_ens_new)
    write_fms_input_files(constraint_u_p_ens_new)
    
    return u_p_ens_new
end



# Perform update
ek_update(iteration_)
