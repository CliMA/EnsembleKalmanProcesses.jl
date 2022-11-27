using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using JLD
using PyPlot
using EnsembleKalmanProcesses

"""
Transfer function 
"""



params_prefix =  "output/output_params_"
input_prefix  =  "output/output_"
output_prefix =  "output/output_"
input_file_template = "input_linear_template"


prior_mean = [0.0; 0.0]
prior_cov  = [1.0 0.0; 0.0 1.0]


N_params, N_y = 2, 1+2
param_bounds = [[nothing, nothing], [nothing, nothing]]


function read_observation(augment::Bool=true)
    obs_mean, obs_noise_cov = [3], 0.1^2*[1.0]
    N_y = length(obs_mean)
    if augment
       return [obs_mean; prior_mean], [obs_noise_cov zeros(Float64, N_y, N_params); zeros(Float64, N_params, N_y) prior_cov]
    else
       return obs_mean, obs_noise_cov
    end
end






function write_solver_input_files(constraint_u_p_ens)

    N_params, N_ens = size(constraint_u_p_ens)

    input_lines = readlines(input_file_template)
    
    modified_lines = [5; 8]
    for i = 1:N_ens
    	constraint_u_p = constraint_u_p_ens[:, i]
 	input_name = input_prefix*string(i)*"/"*"input_file"
        output_name = output_prefix*string(i)*"/"*"output_file.jld"
 
    	input_io = open(input_name, "w")
	 	
    	for (n, line) in enumerate(input_lines)
       	   
	   if n ∉ modified_lines

              write(input_io, line*"\n")

           else
	      if n == 5
	      	 write(input_io, "u = [$(constraint_u_p[1]) ;  $(constraint_u_p[2])] \n")
	      elseif n == 8
	      	 write(input_io, "save(\"$(output_name)\", \"y\" , y) \n")    
	      else
		  @error("STOP: line $(n) should not be modified!")
	      end              
	   
	   end

       end
       close(input_io)
    end
end


function read_solver_output(iteration_::Int64,  ens_index::Int64, augment::Bool=true)
    _, _, _, u_p_ens = read_params( iteration_)
    
    data = load(output_prefix*string(ens_index)*"/"*"output_file.jld")
    
    if augment
       return [data["y"]; u_p_ens[:, ens_index]]
    else 
       return data["y"]
    end

end



function plot(N_iterations::Int64)
	
    u_name = ["θ₁", "θ₂"]
    
    obs_mean, obs_noise_cov = read_observation(false)

    G = [ 1 2]
    u = [ 1; 1]
    y = G*u

    uu_cov_post = inv(G'*(obs_noise_cov\G) + inv(prior_cov))
    u_refs = u_mean_post = prior_mean + uu_cov_post*(G'*(obs_noise_cov\(y - G*prior_mean)))
 
    constraint_u_mean_all, constraint_uu_cov_all = read_all_params(N_iterations)
    constraint_u_mean_all_arr, constraint_uu_cov_all_arr = zeros(N_params, N_iterations+1), zeros(N_params, N_iterations+1)

    for iteration_ = 0: N_iterations

    	constraint_u_mean_all_arr[:, iteration_+1] = constraint_u_mean_all[:, iteration_+1]
	constraint_uu_cov_all_arr[:, iteration_+1] = diag(constraint_uu_cov_all[:,:, iteration_+1])
    end



    ites = Array(0:N_iterations)

    figure(figsize = (7.5, 4.8))
    for i = 1: N_params
        errorbar(ites, constraint_u_mean_all_arr[i,:], yerr=3.0*constraint_uu_cov_all_arr[i,:], fmt="--o",fillstyle="none", label=L"$(u_names[i])")
        semilogy(ites, fill(u_refs[i], N_iterations+1), "--", color="gray")

    end
    xlabel("Iterations")
    legend(bbox_to_anchor=(0.95, 0.8))
    grid("on")
    tight_layout()
    savefig("uki_parameters.pdf")
    close("all")



    uki_errors = zeros(Float64, 2, N_iterations+1)
    for iteration_ = 1:N_iterations+1
        uki_errors[1, iteration_] = norm(constraint_u_mean_all[:,iteration_] - u_mean_post)
        uki_errors[2, iteration_] = norm(constraint_uu_cov_all[:,:,iteration_] - uu_cov_post)
    end

    figure(figsize = (7.5, 4.8))
    semilogy(ites, uki_errors[1, :], "--o", fillstyle="none", label="mean")
    semilogy(ites, uki_errors[2, :], "--o", fillstyle="none", label="cov")
    xlabel("Iterations")
    legend(bbox_to_anchor=(0.95, 0.8))
    grid("on")
    tight_layout()
    savefig("uki_convergence.pdf")
    close("all")

end