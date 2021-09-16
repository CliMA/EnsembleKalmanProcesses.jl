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

function parse_parameters_from_toml(component,parameterization,filename)
    parameter_dict = Dict()
    parameter_parse = TOML.parsefile(filename)[component][parameterization] #note this makes a dictionary of dictionaries
    for (key,val) in parameter_parse
        println(val)
        if haskey(val, "Prior")
            if val["Prior"] != "fixed"
                parameter_dict[key] = Dict("RunValue"       => val["RunValue"],
                                           "Prior"          => val["Prior"],
                                           "Transformation" => val["Transformation"])
            end
        end
    end
    return parameter_dict
end

function parse_function_and_float_args(func_string)
    #parses a function F(x,y,z,...) with x,y,z floats
    function_symbol = Symbol(split(func_string,"(")[1])
    args_string = distribution_arguments = split(split(split(func_string,"(")[2],")")[1],",") # ["0","1"]
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
    else
        function_symbol, distribution_args = parse_function_and_float_args(transformation_string)
        return [function_symbol(distribution_args...)]
    end

end


function create_parameter_distribution(parameter_dict)
    # create the triple
    # (1) parameteter names, (2) distributions, (3) constraints 
    parameter_names = [key for key in keys(parameter_dict)]
    prior_dist = [create_prior(parameter_dict[name]) for name in parameter_names ]
    constraints  = [create_transformation(parameter_dict[name]) for name in parameter_names ]
    println(parameter_names)
    println(prior_dist)
    println(constraints)
    return ParameterDistribution(prior_dist, constraints, parameter_names)
end

function main()

    # parameter file name
    filename = "parameter_file.toml"

    # model component
    component = "atmos"
    parameterization = "edmf"
    # get the parameter info from file
    parameter_dict = parse_parameters_from_toml(component,parameterization,filename)
    # create a parameter distribution
    priors = create_parameter_distribution(parameter_dict)

    # Construct initial ensemble
    N_ens = 10
    initial_params = construct_initial_ensemble(priors, N_ens)

    # then write new parameter files... (10 of them)
    
end

main()
