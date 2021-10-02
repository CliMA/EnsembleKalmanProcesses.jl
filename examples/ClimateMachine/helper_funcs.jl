using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JLD

"""
generate_cm_params(cm_params::Array{Float64}, 
                            cm_param_names::Array{String})

Generate a CLIMAParameters file, setting the values (cp_params) of
a group of CLIMAParameters (cm_param_names).
"""
function generate_cm_params(cm_params::Array{Float64}, cm_param_names::Array{String})
    # Generate version
    version = rand(11111:99999)
    if length(cm_params) == length(cm_param_names)
        open("clima_param_defs_$(version).jl", "w") do io
            for i in 1:length(cm_params)
                write(
                    io,
                    "CLIMAParameters.Atmos.SubgridScale.
              $(cm_param_names[i])(::EarthParameterSet) = 
              $(cm_params[i])\n",
                )
            end
        end
    else
        throw(ArgumentError("Number of parameter names must be equal to number of values provided."))
    end
    return version
end

"""
get_clima_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf::Float64=0.0,
                     z_obs::Union{Array{Float64, 1}, Nothing} = nothing,
                     get_variance::Bool=false)

Get a time-averaged (from ti to tf) vertical profile from a ClimateMachine.jl diagnostics
file, interpolating to observation heights (z_obs) if necessary. If get_variance, return
the variance over the same averaging period.
"""
function get_clima_profile(
    sim_dir::String,
    var_name::Array{String, 1};
    ti::Float64 = 0.0,
    tf::Float64 = 0.0,
    z_obs::Union{Array{Float64, 1}, Nothing} = nothing,
    get_variance::Bool = false,
)

    ds_name = glob("$(sim_dir)/*.nc")[1]
    ds = NCDataset(ds_name)

    if length(var_name) == 1 && occursin("z", var_name[1])
        prof_vec = Array(ds[var_name[1]])
    else
        # Time in Datetime format
        t = Array(ds["time"])
        # Convert to float (seconds)
        t = [(time_ .- t[1]).value / 1000.0 for time_ in t]
        dt = t[2] - t[1]
        ti_diff, ti_index = findmin(broadcast(abs, t .- ti))
        tf_diff, tf_index = findmin(broadcast(abs, t .- tf))

        prof_vec = zeros(0)
        ts_vec = zeros(0, length(ti_index:tf_index))
        # If simulation does not contain values for ti or tf, return high value
        if ti_diff > dt || tf_diff > dt
            for i in 1:length(var_name)
                var_ = Array(ds[var_name[i]])
                append!(prof_vec, 1.0e4 * ones(length(var_[:, 1])))
            end
            println("Initial or final times are not contained in netcdf file.")
        else
            for i in 1:length(var_name)
                var_ = Array(ds[var_name[i]])
                if !isnothing(z_obs)
                    z_ = Array(ds["z"])
                    # Form interpolation
                    var_itp = interpolate((z_, t), var_, (Gridded(Linear()), Gridded(Linear())))
                    # Evaluate interpolation
                    var_ = var_itp(z_obs, t)
                end
                # Average in time
                append!(prof_vec, mean(var_[:, ti_index:tf_index], dims = 2))
                ts_vec = cat(ts_vec, var_[:, ti_index:tf_index], dims = 1)
            end
        end
    end
    close(ds)
    if get_variance
        cov_mat = cov(ts_vec, dims = 2)
        return prof_vec, cov_mat
    else
        return prof_vec
    end
end

"""
agg_clima_ekp(n_params::Integer, output_name::String="ekp_clima")

Aggregate all iterations of the parameter ensembles and write to file, given the
number of parameters in each parameter vector (p).
"""
function agg_clima_ekp(n_params::Integer, output_name::String = "ekp_clima")
    # Get versions
    version_files = glob("versions_*.txt")
    # Recover parameters of last iteration
    last_up_versions = readlines(version_files[end])

    ens_all = Array{Float64, 2}[]
    for (it_num, file) in enumerate(version_files)
        versions = readlines(file)
        u = zeros(length(versions), n_params)
        for (ens_index, version_) in enumerate(versions)
            if it_num == length(version_files)
                open("../../../ClimateMachine.jl/test/Atmos/EDMF/$(version_)", "r") do io
                    u[ens_index, :] =
                        [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index % 3 == 0]
                end
            else
                open("$(version_).output/$(version_)", "r") do io
                    u[ens_index, :] =
                        [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index % 3 == 0]
                end
            end
        end
        push!(ens_all, u)
    end
    save(string(output_name, ".jld"), "ekp_u", ens_all)
    return
end
