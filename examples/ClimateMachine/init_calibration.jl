#=

using Distributions

using ArgParse
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "helper_funcs.jl"))
# Set parameter priors
param_names = ["C_smag", "C_drag"]
n_param = length(param_names)
prior_dist = [ Parameterized(Normal(0.5, 0.05)), Parameterized(Normal(0.001, 0.0001)) ]
constraints = [[no_constraint()], [no_constraint()]]
priors = ParameterDistribution(prior_dist, constraints, param_names)

# Construct initial ensemble
N_ens = 10
initial_params = construct_initial_ensemble(priors, N_ens)
# Generate CLIMAParameters files
params_arr = [row[:] for row in eachrow(initial_params')]
versions = map(param -> generate_cm_params(param, param_names), params_arr)

# Store version identifiers for this ensemble in a common file
open("versions_1.txt", "w") do io
        for version in versions
            write(io, "clima_param_defs_$(version).jl\n")
        end
    end
=#

using Distributions
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

using TOML


function find_in_tags(tags_string,string_list)
    #assume comma separation, this trims whitespace
    tags_list=strip.(split(tags_string,","))
    #returns true only if everything in stringlist occurs somewhere in tags_list
    return all(i in tags_list for i in string_list)
end

function get_tagged_parameters(model_tags,toml_dict)
    parameter_dict = Dict()

    # function returning true if all model_tags are in the current parameter 
    for (key,val) in toml_dict
        if haskey(val,"Tags")
            if find_in_tags(val["Tags"],model_tags)
                if val["Prior"] != "fixed"
                    #prefix = component*"."*parameterization*"."
                    parameter_dict[key] = Dict("Tags"           => val["Tags"],
                                               "RunValue"       => val["RunValue"],
                                               "Prior"          => val["Prior"],
                                               "Transformation" => val["Transformation"])
                end
            end
        end
    end
    return parameter_dict
end

function parse_function_and_float_args(func_string)
    #parses a function F(x,y,z,...) with x,y,z floats
    function_symbol = Symbol(split(func_string,"(")[1])
    args_string = split(split(split(func_string,"(")[2],")")[1],",") # ["0","1"]
    args =  parse.(Float64,args_string)
    return function_symbol, args
end

function create_prior(parameter)
    # if the distribution is a Julia distribution type
    # distn(a,b,...,e) where a,...,e are just floats
    
    prior_string = parameter["Prior"] #"Normal(0,1)"
    function_symbol, distribution_args = parse_function_and_float_args(prior_string)
    distribution_function = getfield(Distributions, function_symbol) # function: Normal

    return Parameterized(distribution_function(distribution_args...))
end

function create_transformation(parameter)

    transformation_string = parameter["Transformation"]
    if transformation_string == "none"
        return [no_constraint()]
    else #if it is a different predefined constraint
        function_symbol, distribution_args = parse_function_and_float_args(transformation_string)
        transform_function = getfield(ParameterDistributionStorage, function_symbol) 
       
        return [transform_function(distribution_args...)]
    end

end


function create_parameter_distribution(parameter_dict)
    # create the triple
    # (1) parameteter names, (2) distributions, (3) constraints 
    parameter_names = [key for key in keys(parameter_dict)]
    prior_dist = [create_prior(parameter_dict[name]) for name in parameter_names ]
    constraints  = [create_transformation(parameter_dict[name]) for name in parameter_names ]
    return ParameterDistribution(prior_dist, constraints, parameter_names)
end

function create_dict_from_ensemble(parameter_dict, initial_params, param_names)
    param_dict_ensemble=[]
    for param_set in eachrow(initial_params') # for each parameter realization
        #overwrite the param set
        for (idx,name) in enumerate(param_names)
            parameter_dict[name]["RunValue"] = param_set[idx]
        end
        push!(param_dict_ensemble,deepcopy(parameter_dict))
    end
    return param_dict_ensemble
end

function write_toml_ensemble(ens_dir_name,filename,param_dict_ensemble)
    if ~isdir(ens_dir_name)
        mkdir(ens_dir_name)
    else
        println("overwriting files in ", ens_dir_name)
        rm(ens_dir_name,recursive=true)
        mkdir(ens_dir_name)
        
    end
    for (idx,param_dict) in enumerate(param_dict_ensemble)
        member_filename = ens_dir_name*"/member_"*string(idx)*".toml" #toml file
        io_member_file=open(member_filename, "a")
        param_names = [ "["*string(key)*"]" for (key,val) in param_dict]
        param_vals = ["RunValue = "*string(param_dict[key]["RunValue"])*"\n" for (key,val) in param_dict]
        open(filename) do io
            #gets the right toml name
            change_value=[Int(0)]
            change_value_flag=[false]
            for line in eachline(io, keep=true)
                for (name_idx,name) in enumerate(param_names)
                    if contains(line, name) #found the parameter
                        # change the next RunValue instance
                        change_value[1] = name_idx
                        change_value_flag[1] = true
                    end
                    # if we have flagged a parameter value to be changed, we change it and then remove the flag.
                    if change_value_flag[1] == true
                        if contains(line, "RunValue") #found the parameter
                            line = param_vals[change_value[1]]
                            change_value_flag[1]=false
                        end
                    end
                end
                write(io_member_file,line) #write the line to new file
            end
            
        end
        close(io_member_file)
    end
    
end


function main()

    # parameter file name
    filename = "parameter_file.toml"
    expname = "test"
    
    # model component and parameterization choice
    model_tags = ["atmos","edmf"]
    
    #directory name
    ens_dir_name = expname*"_1"
    # get the parameter info from file
    toml_dict = TOML.parsefile(filename)
    parameter_dict = get_tagged_parameters(model_tags, toml_dict)
    # create a parameter distribution
    # parameter_dict, also contains the naming in the toml file as the ["Prefix"]
    priors = create_parameter_distribution(parameter_dict)

    # Construct initial ensemble
    N_ens = 10
    initial_unconstrained_params = construct_initial_ensemble(priors, N_ens)
    initial_constrained_params = transform_unconstrained_to_constrained(priors,initial_unconstrained_params)
    param_dict_ensemble = create_dict_from_ensemble(parameter_dict,initial_constrained_params, get_name(priors))

    # write the parameter files, currently using sed-like insertions
    write_toml_ensemble(ens_dir_name, filename, param_dict_ensemble)

    println("Created ", N_ens, " files, in directory ",ens_dir_name)
    
end

main()
