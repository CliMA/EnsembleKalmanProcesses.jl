using Distributions
using JLD2
using ArgParse
using NCDatasets
using LinearAlgebra
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

using TOML
include(joinpath(@__DIR__, "toml_utils.jl"))
include(joinpath(@__DIR__, "helper_funcs.jl"))
#=
"""
ek_update(iteration_::Int64)

Update CLIMAParameters ensemble using Ensemble Kalman Inversion,
return the ensemble and the parameter names.
"""
function ek_update(iteration_::Int64)
    # Recover versions from last iteration
    versions = readlines("versions_$(iteration_).txt")
    n_params = 0
    u_names = String[]
    open("$(versions[1]).output/$(versions[1])", "r") do io
        append!(u_names, [strip(string(split(line, "(")[1]), [' ']) for (index, line) in enumerate(eachline(io)) if index%3 == 2])
        n_params = length(u_names)
    end
    # Recover ensemble from last iteration, [N_ens, N_params]
    u = zeros(length(versions), n_params)
    for (ens_index, version_) in enumerate(versions)
        open("$(version_).output/$(version_)", "r") do io
            u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
        end
    end
    u = Array(u')

    # Set averaging period for loss function
    t0_ = 0.0
    tf_ = 1800.0
    # Get observations (CliMA)
    yt = zeros(0)
    yt_var_list = []
    y_names = ["u", "v"]
    yt_, yt_var_ = get_clima_profile("truth_output", y_names, ti=t0_, tf=tf_, get_variance=true)
    # Add nugget to variance (regularization)
    yt_var_ = yt_var_ + Matrix(0.1I, size(yt_var_)[1], size(yt_var_)[2])
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)

    # Get outputs
    g_names = y_names
    g_ =  get_clima_profile("$(versions[1]).output", g_names, ti=t0_, tf=tf_)
    g_ens = zeros(length(versions), length(g_))
    for (ens_index, version) in enumerate(versions)
        g_ens[ens_index, :] = get_clima_profile("$(version).output", g_names, ti=t0_, tf=tf_)
    end
    g_ens = Array(g_ens')
    # Construct EKP
    ekobj = EnsembleKalmanProcess(u, yt_, yt_var_, Inversion())
    # Advance EKP
    update_ensemble!(ekobj, g_ens)
    # Get new step
    u_new = get_u_final(ekobj)
    return u_new, u_names
end
=#

function construct_output_ensemble(y_names,t0_,tf_,N_ens)
    # Get outputs
    g_names = y_names
    g_ =  get_clima_profile("$(versions[1]).output", g_names, ti=t0_, tf=tf_)
    g_ens = zeros(N_ens, length(g_))
    for (ens_index, version) in enumerate(versions)
        g_ens[ens_index, :] = get_clima_profile("$(version).output", g_names, ti=t0_, tf=tf_)
    end
    return Array(g_ens') #we call this g_ens

end

              
function main()

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
    
    # file names
    expname = "test"
    in_dir_name = expname*"_"*String(iteration_)
    out_dir_name = expname*"_"*String(iteration_+1)

    # to read ensemble in from old param files (don't need to do)
    #toml_dict_dict, parameter_dict_dict, N_ens = read_ensemble_from_dir(ens_dir_name)
    
    #load ekp and priors
    @load in_dir_name*"/priors.jld2" priors
    @load in_dir_name*"/ekobj.jld2" ekobj
    
    #get output data
    N_ens = ekobj.N_ens
    #magic sim numbers
    t0_ = 0.0
    tf_ = 1800.0
    y_names = ["u", "v"]
    g_ens = construct_output_ensemble(y_names,t0_,tf_,N_ens)
    
    # Perform update
    update_ensemble!(ekobj, g_ens)
    unconstrained_params = get_u_final(ekobj)
    constrained_params = hcat([transform_unconstrained_to_constrained(priors,unconstrained_params[:,i]) for i in collect(1:N_ens)]...)

    #load parameter file templates
    @load out_dir_name*"/toml_dict.jld2" toml_dict
    @load out_dir_name*"/parameter_dict.jld2" parameter_dict
    param_dict_ensemble = create_dict_from_ensemble(parameter_dict, constrained_params, get_name(priors))
    write_toml_ensemble_from_dict(out_dir_name, toml_dict, param_dict_ensemble)
    println("Created ", N_ens, " files, in directory ", out_dir_name)

    #save ekobj,priors, and template files
    @save out_dir_name*"/priors.jld2" priors
    @save out_dir_name*"/ekobj.jld2" ekobj
    @save out_dir_name*"/toml_dict.jld2" toml_dict
    @save out_dir_name*"/parameter_dict.jld2" parameter_dict

end
main()
