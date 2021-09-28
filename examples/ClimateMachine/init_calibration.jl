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

using JLD2
using TOML
include(joinpath(@__DIR__, "toml_utils.jl")) #read/write files
include(joinpath(@__DIR__, "helper_funcs.jl")) #access clima data

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
    #assume predefined constraint for now
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
    N_ens = 100
    initial_unconstrained_params = construct_initial_ensemble(priors, N_ens)
    initial_constrained_params = hcat([transform_unconstrained_to_constrained(priors,initial_unconstrained_params[:,i]) for i in collect(1:N_ens)]...)
    param_dict_ensemble = create_dict_from_ensemble(parameter_dict,initial_constrained_params, get_name(priors))
    
    # write the parameter files, currently using sed-like insertions
    #write_toml_ensemble(ens_dir_name, filename, param_dict_ensemble)

    # or write the parameter files by adding to the dict, and writing back to file
    # note order is not preserved
    write_toml_ensemble_from_dict(ens_dir_name, toml_dict, param_dict_ensemble)

    println("Created ", N_ens, " files, in directory ",ens_dir_name)

    #define the initial EKP object
    # requires: the initial ensemble
    #           data and variability
    #some magic numbers 
    t0_ = 0.0
    tf_ = 1800.0
    y_names = ["u", "v"]
    
    yt = zeros(0)
    yt_var_list = []
    yt_, yt_var_ = get_clima_profile("truth_output", y_names, ti=t0_, tf=tf_, get_variance=true)
    # Add nugget to variance (regularization)
    yt_var_ = yt_var_ + Matrix(0.1I, size(yt_var_)[1], size(yt_var_)[2])
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)

    ekobj = EnsembleKalmanProcess(initial_constrained_params, yt_, yt_var_, Inversion())
    @save ens_dir_name*"/priors.jld2" priors
    @save ens_dir_name*"/ekobj.jld2" ekobj  
    @save ens_dir_name*"/toml_dict.jld2" toml_dict
    @save ens_dir_name*"/parameter_dict.jld2" parameter_dict
end

main()
