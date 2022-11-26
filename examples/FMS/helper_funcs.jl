using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using JLD

"""
Transfer function 
"""



params_prefix =  "output/output_params_"
input_prefix  =  "output/output"
output_prefix =  "output/output"
N_params = 2
param_bounds = [[nothing, nothing], [nothing, nothing]]

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


function constraint(u::Array{Float64, 1})
  constraint_u = similar(u)
  for (i, u_i) in enumerate(u)
      constraint_u[i] = constraint(u_i, param_bounds[i][1], param_bounds[i][2])
  end
  return constraint_u
end


function constraint(u_ens::Array{Float64, 2})
  constraint_u_ens = similar(u_ens)
  N_params, N_ens = size(u_ens)

  for i = 1:N
      constraint_u_ens[:, i] = constraint_helper(u_ens[:, i])
  end

  return constraint_u_ens
end


# save mean and covariance and u_ens
function save_params(ukiobj, iteration_::Int64)
    mean, cov = ukiobj.process.u_mean[end], ukiobj.process.uu_cov[end]
    u_ens = construct_sigma_ensemble(ukiobj.process, mean, cov)
    u_p_ens = get_u_final(ukobj)
         
    save(params_prefix*string(iteration_)*"jld", "mean", mean, "cov", cov, "u_ens", u_ens, "u_p_ens", u_p_ens)	 
end


function read_params( iteration_::Int64)
   data = load(params_prefix*string(iteration_)*"jld")
   
   return data["mean"], data["cov"], data["u_ens"], data["u_p_ens"]
end



function write_fms_input_files(constraint_u_p_ens)

    N_params, N_ens = size(constraint_u_p_ens)

    input_lines = readlines(input_file_template)
    
    for i = 1:N_ens
    	constraint_u_p = constraint_u_p_ens[:, i]
 	input_name = input_prefix*string(i)*"/"*"input_file"
        output_name = output_prefix*string(i)*"/"*"output_file"
 
    	input_io = open(input_name, "w")
 	
    	for (n, line) in enumerate(input_lines)
       	   
	   if n ∉ modified_lines

              write(input_io, line)

           else
	      if n == 5
	      	 write(input_io, "u = [$(constraint_u_p[1])  $(constraint_u_p[2])]")
	      elseif n == 8
	      	 write(input_io, "save(\"$(output_name)\", \"y\" , y)")    
	      else
		  @error("STOP: line $(n) should not be modified!")
	      end              
	   
	   end

       end

    end

    close(input_io);

end


function read_fms_output(iteration_::Int64,  ens_index::Int64)
    _, _, _, u_p_ens = read_params( iteration_)
    
    data = load(output_prefix*string(ens_index)*"/"*"output")
    
    return [data["y"]; u_p_ens[:, ens_index]]
end


