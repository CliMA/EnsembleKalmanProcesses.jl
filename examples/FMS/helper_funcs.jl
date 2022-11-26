using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using JLD
using EnsembleKalmanProcesses

"""
Transfer function 
"""



params_prefix =  "output/output_params_"
input_prefix  =  "output/output_"
output_prefix =  "output/output_"
input_file_template = "linear.jl"

u_name = ["theta", "theta"]

prior_mean = [0.0; 0.0]
prior_cov  = [1.0 0.0; 0.0 1.0]


obs_mean, obs_noise_cov = [3], 0.1^2*[1.0]

G = [ 1 2]
u = [ 1; 1]
y = G*u
uu_cov_post = inv(G'*(obs_noise_cov\G) + inv(prior_cov))

u_refs = u_mean_post = prior_mean + uu_cov_post*(G'*(obs_noise_cov\(y - G*prior_mean)))





N_params, N_y = 2, 1+2
param_bounds = [[nothing, nothing], [nothing, nothing]]





α_reg = 1.0 
update_freq = 1 
sigma_points_type = "symmetric"

N_ens = 2N_params  + 1


function constraint(u::Float64, u_low, u_up)
    if(isnothing(u_low) && isnothing(u_up))
    	return u
    elseif (isnothing(u_low)  &&  !isnothing(u_up))
        return u_up - exp(u)
    elseif (!isnothing(u_low)  &&  isnothing(u_up))	
        return u_low + exp(u)
    else
        return u_low + (u_up - u_low)/(1 + exp(u))
    end
end

function dconstraint(u::Float64, u_low, u_up)
    if(isnothing(u_low) && isnothing(u_up))
        return 1
    elseif (isnothing(u_low)  &&  !isnothing(u_up))
    	return -exp(u)
    elseif (!isnothing(u_low)  &&  isnothing(u_up))
        return exp(u)
    else
        return  -(u_up - u_low)*exp(u)/(1 + exp(u))^2
    end
end

function constraint(u::Array{Float64, 1})
  constraint_u = similar(u)
  for (i, u_i) in enumerate(u)
      constraint_u[i] = constraint(u_i, param_bounds[i][1], param_bounds[i][2])
  end
  return constraint_u
end

function constraint(u_mean::Array{Float64, 1}, uu_cov::Array{Float64, 2})
  N_params = length(u_mean)
  dc = zeros(N_params, N_params)
  for i = 1:N_params
      dc[i,i] = dconstraint(u_mean[i], param_bounds[i][1], param_bounds[i][2])
  end
  
  constraint_uu_cov = dc * uu_cov * dc'
  return constraint_uu_cov
end

function constraint(u_ens::Array{Float64, 2})
  constraint_u_ens = similar(u_ens)
  N_params, N_ens = size(u_ens)

  for i = 1:N_ens
      constraint_u_ens[:, i] = constraint(u_ens[:, i])
  end

  return constraint_u_ens
end


# save mean and covariance and u_ens
function save_params(ukiobj, iteration_::Int64)
    mean, cov = ukiobj.process.u_mean[end], ukiobj.process.uu_cov[end]
    u_ens = EnsembleKalmanProcesses.construct_sigma_ensemble(ukiobj.process, mean, cov)
    u_p_ens = get_u_final(ukiobj)
         
    save(params_prefix*string(iteration_)*".jld", "mean", mean, "cov", cov, "u_ens", u_ens, "u_p_ens", u_p_ens)	 
end


function read_params( iteration_::Int64)
  
   data = load(params_prefix*string(iteration_)*".jld")
   
   return data["mean"], data["cov"], data["u_ens"], data["u_p_ens"]

end


function read_all_params(N_iterations::Int64)
    constraint_u_mean_all = zeros(N_params, N_iterations + 1)
    constraint_uu_cov_all = zeros(N_params, N_params, N_iterations + 1)

    for iteration_ = 0:N_iterations
        data = load(params_prefix*string(iteration_)*".jld")
	u_mean, uu_cov = data["mean"], data["cov"]
 	
	constraint_u_mean_all[:, iteration_+1] = constraint(u_mean)
        constraint_uu_cov_all[:, :, iteration_+1] = constraint(u_mean, uu_cov)
	
    end

    return constraint_u_mean_all, constraint_uu_cov_all
end

function read_observation()

    obs_mean, obs_noise_cov = [3], 0.1^2*[1.0]
    N_y = length(obs_mean)
    return [obs_mean; prior_mean], [obs_noise_cov zeros(Float64, N_y, N_params); zeros(Float64, N_params, N_y) prior_cov]
end

function write_fms_input_files(constraint_u_p_ens)

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


function read_fms_output(iteration_::Int64,  ens_index::Int64)
    _, _, _, u_p_ens = read_params( iteration_)
    
    data = load(output_prefix*string(ens_index)*"/"*"output_file.jld")
    
    return [data["y"]; u_p_ens[:, ens_index]]
end


