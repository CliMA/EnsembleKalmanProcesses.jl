using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JLD

function run_SCAMPy(u::Array{FT, 1},
                    u_names::Array{String, 1},
                    y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
                    scm_dir::String,
                    ti::Union{FT, Array{FT,1}},
                    tf::Union{FT, Array{FT,1}, Nothing} = nothing,
                    ) where {FT<:AbstractFloat}

    # Check dimensionality
    @assert length(u_names) == length(u)
    exe_path = string(scm_dir, "call_SCAMPy.sh")
    sim_uuid  = u[1]
    for i in 2:length(u_names)
        sim_uuid = string(sim_uuid,u[i])
    end
    command = `bash $exe_path $u $u_names`
    run(command)

    # SCAMPy file descriptor
    sim_uuid = string(sim_uuid, ".txt")
    sim_dirs = readlines(sim_uuid)
    run(`rm $sim_uuid`)
    
    y_scm = zeros(0)
    # For now it is assumed that if these do not coincide,
    # there is only one simulation.
    if length(ti) != length(sim_dirs)
        @assert length(sim_dirs) == 1
        sim_dir = sim_dirs[1]
        for i in 1:length(ti)
            ti_ = ti[i]
            if !isnothing(tf)
                tf_ = tf[i]
            else
                tf_ = tf
            end
            if typeof(y_names)==Array{Array{String,1},1}
                y_names_ = y_names[i]
            else
                y_names_ = y_names
            end
            append!(y_scm, get_profile(sim_dir, y_names_, ti = ti_, tf = tf_))
        end
        run(`rm -r $sim_dir`)
    else
        for i in 1:length(sim_dirs)
            sim_dir = sim_dirs[i]
            if length(ti) > 1
                ti_ = ti[i]
                if !isnothing(tf)
                    tf_ = tf[i]
                else
                    tf_ = tf
                end
            else
                ti_ = ti
                tf_ = tf
            end

            if typeof(y_names)==Array{Array{String,1},1}
                y_names_ = y_names[i]
            else
                y_names_ = y_names
            end
            append!(y_scm, get_profile(sim_dir, y_names_, ti = ti_, tf = tf_))
            run(`rm -r $sim_dir`)
        end
    end

    for i in eachindex(y_scm)
        if isnan(y_scm[i])
            y_scm[i] = 1.0e5
        end
    end

    return y_scm
end

function obs_LES(y_names::Array{String, 1},
                    sim_dir::String,
                    ti::Float64,
                    tf::Float64;
                    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                    ) where {FT<:AbstractFloat}
    
    y_names_les = get_les_names(y_names, sim_dir)
    y_highres = get_profile(sim_dir, y_names_les, ti = ti, tf = tf)
    y_tvar = get_timevar_profile(sim_dir, y_names_les,
        ti = ti, tf = tf, z_scm=z_scm)
    if !isnothing(z_scm)
        y_ = zeros(0)
        z_les = get_profile(sim_dir, ["z_half"])
        num_outputs = Integer(length(y_highres)/length(z_les))
        for i in 1:num_outputs
            y_itp = interpolate( (z_les,), 
                y_highres[1 + length(z_les)*(i-1) : i*length(z_les)],
                Gridded(Linear()) )
            append!(y_, y_itp(z_scm))
        end
    else
        y_ = y_highres
    end
    return y_, y_tvar
end

function interp_padeops(padeops_data,
                    padeops_z,
                    padeops_t,
                    z_scm,
                    t_scm
                    )
    # Weak verification of limits for independent vars 
    @assert abs(padeops_z[end] - z_scm[end])/padeops_z[end] <= 0.1
    @assert abs(padeops_z[end] - z_scm[end])/z_scm[end] <= 0.1

    # Create interpolating function
    padeops_itp = interpolate( (padeops_t, padeops_z), padeops_data,
                ( Gridded(Linear()), Gridded(Linear()) ) )
    return padeops_itp(t_scm, z_scm)
end

function padeops_m_σ2(padeops_data,
                    padeops_z,
                    padeops_t,
                    z_scm,
                    t_scm,
                    dims_ = 1)
    padeops_snapshot = interp_padeops(padeops_data,padeops_z,padeops_t,z_scm, t_scm)
    # Compute variance along axis dims_
    padeops_var = cov(padeops_data, dims=dims_)
    return padeops_snapshot, padeops_var
end

function get_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=nothing,
                     getFullHeights=false)

    if length(var_name) == 1 && occursin("z_half", var_name[1])
        prof_vec = nc_fetch(sim_dir, "profiles", var_name[1])
    else
        t = nc_fetch(sim_dir, "timeseries", "t")
        dt = length(t) > 1 ? abs(t[2]-t[1]) : 0.0
        # Check that times are contained in simulation output
        ti_diff, ti_index = findmin( broadcast(abs, t.-ti) )
        if !isnothing(tf)
            tf_diff, tf_index = findmin( broadcast(abs, t.-tf) )
        end
        prof_vec = zeros(0)
        # If simulation does not contain values for ti or tf, return high value
        if ti_diff > dt
            println("ti_diff > dt ", "ti_diff = ", ti_diff, "dt = ", dt, "ti = ", ti,
                 "t[1] = ", t[1], "t[end] = ", t[end])
            for i in 1:length(var_name)
                var_ = nc_fetch(sim_dir, "profiles", "z_half")
                append!(prof_vec, 1.0e5*ones(length(var_[:])))
            end
        else
            for i in 1:length(var_name)
                if occursin("horizontal_vel", var_name[i])
                    u_ = nc_fetch(sim_dir, "profiles", "u_mean")
                    v_ = nc_fetch(sim_dir, "profiles", "v_mean")
                    var_ = sqrt.(u_.^2 + v_.^2)
                else
                    var_ = nc_fetch(sim_dir, "profiles", var_name[i])
                    # LES vertical fluxes are per volume, not mass
                    if occursin("resolved_z_flux", var_name[i])
                        rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
                        var_ = var_.*rho_half
                    end
                end
                if !isnothing(tf)
                    append!(prof_vec, mean(var_[:, ti_index:tf_index], dims=2))
                else
                    append!(prof_vec, var_[:, ti_index])
                end
            end
        end
    end
    return prof_vec 
end

function get_timevar_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=0.0,
                     getFullHeights=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                     var_cond = False)

    t = nc_fetch(sim_dir, "timeseries", "t")
    dt = t[2]-t[1]
    ti_diff, ti_index = findmin( broadcast(abs, t.-ti) )
    tf_diff, tf_index = findmin( broadcast(abs, t.-tf) )
    prof_vec = zeros(0, length(ti_index:tf_index))

    for i in 1:length(var_name)
        var_ = nc_fetch(sim_dir, "profiles", var_name[i])
        # LES vertical fluxes are per volume, not mass
        if occursin("resolved_z_flux", var_name[i])
            rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
            var_ = var_.*rho_half
        end
        prof_vec = cat(prof_vec, var_[:, ti_index:tf_index], dims=1)
    end
    if !isnothing(z_scm)
        if !getFullHeights
            z_les = get_profile(sim_dir, ["z_half"])
        else
            z_les = get_profile(sim_dir, ["z"])
        end
        num_outputs = Integer(length(prof_vec[:, 1])/length(z_les))
        prof_vec_zscm = zeros(0, length(ti_index:tf_index))
        maxvar_vec = zeros(num_outputs) 
        for i in 1:num_outputs
            maxvar_vec[i] = maximum(var(prof_vec[1 + length(z_les)*(i-1) : i*length(z_les), :], dims=2))
            prof_vec_itp = interpolate( (z_les, 1:tf_index-ti_index+1),
                prof_vec[1 + length(z_les)*(i-1) : i*length(z_les), :],
                ( Gridded(Linear()), NoInterp() ))
            prof_vec_zscm = cat(prof_vec_zscm,
                prof_vec_itp(z_scm, 1:tf_index-ti_index+1), dims=1)
        end

        cov_mat = cov(prof_vec_zscm, dims=2)
        if var_cond
            for i in 1:num_outputs
                #cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), :] = (
                #    cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), :] .* sqrt(maxvar_vec[i]) )
                #cov_mat[:, 1 + length(z_scm)*(i-1) : i*length(z_scm)] = (
                #    cov_mat[:, 1 + length(z_scm)*(i-1) : i*length(z_scm)] .* sqrt(maxvar_vec[i]) )
                cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)] = (
                   cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)]
                   - Diagonal(cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)])
                   + maxvar_vec[i]*Diagonal(ones(length(z_scm), length(z_scm))))
            end
        end
    else
        cov_mat = cov(prof_vec, dims=2)
    end
    return cov_mat
end

function get_les_names(scm_y_names::Array{String,1}, sim_dir::String)
    y_names = deepcopy(scm_y_names)
    if "thetal_mean" in y_names
        if occursin("GABLS",sim_dir) || occursin("Soares",sim_dir)
            y_names[findall(x->x=="thetal_mean", y_names)] .= "theta_mean"
        else
            y_names[findall(x->x=="thetal_mean", y_names)] .= "thetali_mean"
        end
    end
    if "total_flux_qt" in y_names
        y_names[findall(x->x=="total_flux_qt", y_names)] .= "resolved_z_flux_qt"
    end
    if "total_flux_h" in y_names && (occursin("GABLS",sim_dir) || occursin("Soares",sim_dir))
        y_names[findall(x->x=="total_flux_h", y_names)] .= "resolved_z_flux_theta"
    elseif "total_flux_h" in y_names
        y_names[findall(x->x=="total_flux_h", y_names)] .= "resolved_z_flux_thetali"
    end
    if "u_mean" in y_names
        y_names[findall(x->x=="u_mean", y_names)] .= "u_translational_mean"
    end
    if "v_mean" in y_names
        y_names[findall(x->x=="v_mean", y_names)] .= "v_translational_mean"
    end
    if "tke_mean" in y_names
        y_names[findall(x->x=="tke_mean", y_names)] .= "tke_nd_mean"
    end
    return y_names
end

function nc_fetch(dir, nc_group, var_name)
    find_prev_to_name(x) = occursin("Output", x)
    split_dir = split(dir, ".")
    sim_name = split_dir[findall(find_prev_to_name, split_dir)[1]+1]
    ds = NCDataset(string(dir, "/stats/Stats.", sim_name, ".nc"))
    ds_group = ds.group[nc_group]
    ds_var = deepcopy( Array(ds_group[var_name]) )
    close(ds)
    return Array(ds_var)
end

"""
agg_clima_ekp(n_params::Integer, output_name::String="ekp_clima")

Aggregate all iterations of the parameter ensembles and write to file.
"""
function agg_clima_ekp(n_params::Integer, output_name::String="ekp_clima")
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
                    u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
                end
            else
                open("$(version_).output/$(version_)", "r") do io
                    u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
                end
            end
        end
        push!(ens_all, u)
    end
    save( string(output_name,".jld"), "ekp_u", ens_all)
    return
end

"""
    precondition_ensemble!(params::Array{FT, 2}, priors, 
        param_names::Vector{String}, ::Union{Array{String, 1}, Array{Array{String,1},1}}, 
        ti::Union{FT, Array{FT,1}}, tf::Union{FT, Array{FT,1}};
        lim::FT=1.0e3,) where {IT<:Int, FT}

Substitute all unstable parameters by stable parameters drawn from 
the same prior.
"""
function precondition_ensemble!(params::Array{FT, 2}, priors,
    param_names::Vector{String}, y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
    ti::Union{FT, Array{FT,1}};
    tf::Union{FT, Array{FT,1}, Nothing}=nothing, lim::FT=1.0e4,) where {IT<:Int, FT}

    # Check dimensionality
    @assert length(param_names) == size(params, 1)

    scm_dir = "/home/ilopezgo/SCAMPy/"
    params_i = deepcopy(exp.(params))
    g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names, y_names, scm_dir, ti, tf)
    params_i = [row[:] for row in eachrow(params_i')]
    N_ens = size(params_i, 1)
    g_ens_arr = pmap(g_, params_i) # [N_ens N_output]
    @assert size(g_ens_arr, 1) == N_ens
    N_out = size(g_ens_arr, 2)
    # If more than 1/4 of outputs are over limit lim, deemed as unstable simulation
    unstable_point_inds = findall(x->x>Integer(floor(N_out/4)), count.(x->x>lim, g_ens_arr))
    println(string("Unstable parameter indices: ", unstable_point_inds))
    # Recursively eliminate all unstable parameters
    if !isempty(unstable_point_inds)
        println(string(length(unstable_point_inds), " unstable parameters found.
            Sampling new parameters from prior." ))
        new_params = construct_initial_ensemble(priors, length(unstable_point_inds))
        precondition_ensemble!(new_params, priors, param_names,
            y_names, ti, tf=tf, lim=lim)
        params[:, unstable_point_inds] = new_params
    end
    println("\nPreconditioning finished.")
    return
end

"""
    logmean_and_logstd(μ, σ)

Returns the lognormal parameters μ and σ from the mean μ and std σ of the 
lognormal distribution.
"""
function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2/μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2/μ^2)))
    return μ_log, σ_log
end

"""
    logmean_and_logstd_from_mode_std(μ, σ)

Returns the lognormal parameters μ and σ from the mode and the std σ of the 
lognormal distribution.
"""
function logmean_and_logstd_from_mode_std(mode, σ)
    σ_log = sqrt( log(mode)*(log(σ^2)-1.0)/(2.0-log(σ^2)*3.0/2.0) )
    μ_log = log(mode) * (1.0-log(σ^2)/2.0)/(2.0-log(σ^2)*3.0/2.0)
    return μ_log, σ_log
end

"""
    mean_and_std_from_ln(μ, σ)

Returns the mean and variance of the lognormal distribution
from the lognormal parameters μ and σ.
"""
function mean_and_std_from_ln(μ_log, σ_log)
    μ = exp(μ_log + σ_log^2/2)
    σ = sqrt( (exp(σ_log^2) - 1)* exp(2*μ_log + σ_log^2) )
    return μ, σ
end

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)
