using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JLD
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage

"""
    run_SCAMPy(u, u_names, y_names, scm_dir,
                    ti, tf = nothing;
                    norm_var_list = nothing,
                    P_pca_list = nothing)

Run call_SCAMPy.sh using a set of parameters u and return
the value of outputs defined in y_names, possibly after
normalization and projection onto lower dimensional space
using PCA.

Inputs:
 - u :: Values of parameters to be used in simulations.
 - u_names :: SCAMPy names for parameters u.
 - y_names :: Name of outputs requested for each flow configuration.
 - ti :: Vector of starting times for observation intervals. If tf=nothing,
         snapshots at ti are returned.
 - tf :: Vector of ending times for observation intervals.
 - norm_var_list :: Pooled variance vectors. If given, use to normalize output.
 - P_pca_list :: Vector of projection matrices P_pca for each flow configuration.
Outputs:
 - g_scm :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca :: Projection of g_scm onto principal subspace spanned by eigenvectors.
"""
function run_SCAMPy(u::Array{FT, 1},
                    u_names::Array{String, 1},
                    y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
                    scm_dir::String,
                    ti::Union{FT, Array{FT,1}},
                    tf::Union{FT, Array{FT,1}, Nothing} = nothing;
                    norm_var_list = nothing,
                    P_pca_list = nothing,
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
    
    g_scm = zeros(0)
    g_scm_pca = zeros(0)
    # For now it is assumed that if length(ti) != length(sim_dirs),
    # there is only one simulation and multiple time intervals.
    if length(ti) != length(sim_dirs)
        @assert length(sim_dirs) == 1
        sim_dir = sim_dirs[1]
        for (i, ti_) in enumerate(ti)
            tf_ = !isnothing(tf) ? tf[i] : nothing
            y_names_ = typeof(y_names)==Array{Array{String,1},1} ? y_names[i] : y_names

            g_scm_flow = get_profile(sim_dir, y_names_, ti = ti_, tf = tf_)
            if !isnothing(norm_var_list)
                g_scm_flow = normalize_profile(g_scm_flow, y_names_, norm_var_list[1])
            end
            if !isnothing(P_pca_list)
                append!(g_scm_pca, P_pca_list[1]' * g_scm_flow)
            end
            append!(g_scm, g_scm_flow)
        end
        run(`rm -r $sim_dir`)
    else
        for (i, sim_dir) in enumerate(sim_dirs)
            ti_ = ti[i]
            tf_ = !isnothing(tf) ? tf[i] : nothing
            y_names_ = typeof(y_names)==Array{Array{String,1},1} ? y_names[i] : y_names

            g_scm_flow = get_profile(sim_dir, y_names_, ti = ti_, tf = tf_)
            if !isnothing(norm_var_list)
                g_scm_flow = normalize_profile(g_scm_flow, y_names_, norm_var_list[i])
            end
            append!(g_scm, g_scm_flow)
            if !isnothing(P_pca_list)
                append!(g_scm_pca, P_pca_list[i]' * g_scm_flow)
            end
            
            run(`rm -r $sim_dir`)
        end
    end

    for i in eachindex(g_scm)
        if isnan(g_scm[i])
            g_scm[i] = 1.0e5
        end
    end
    if !isnothing(P_pca_list)
        for i in eachindex(g_scm_pca)
            if isnan(g_scm_pca[i])
                g_scm_pca[i] = 1.0e5
            end
        end
        println("LENGTH OF G_SCM_ARR", length(g_scm))
        println("LENGTH OF G_SCM_ARR_PCA", length(g_scm_pca))
        return g_scm, g_scm_pca
    else
        return g_scm
    end
end

"""
    obs_LES(y_names, sim_dir, ti, tf;
            z_scm = nothing, normalize = false)

Get LES output for observed variables y_names, interpolated to
z_scm (if given), and possibly normalized with respect to the pooled variance.

Inputs:
 - y_names :: Name of outputs requested for each flow configuration.
 - ti :: Vector of starting times for observation intervals. If tf=nothing,
         snapshots at ti are returned.
 - tf :: Vector of ending times for observation intervals.
 - z_scm :: If given, interpolate LES observations to given levels.
 - normalize :: If true, normalize observations and cov matrix by pooled variances.
Outputs:
 - y_ :: Mean of observations, possibly interpolated to z_scm levels.
 - y_tvar :: Observational covariance matrix, possibly pool-normalized.
"""
function obs_LES(y_names::Array{String, 1},
                    sim_dir::String,
                    ti::Float64,
                    tf::Float64;
                    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                    normalize = false,
                    ) where {FT<:AbstractFloat}
    
    y_names_les = get_les_names(y_names, sim_dir)
    y_tvar, maxvar_vec = get_timevar_profile(sim_dir, y_names_les,
        ti = ti, tf = tf, z_scm=z_scm, normalize=normalize)
    y_highres = get_profile(sim_dir, y_names_les, ti = ti, tf = tf)
    if normalize
        y_highres = normalize_profile(y_highres, y_names, maxvar_vec)
    end
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
    return y_, y_tvar, maxvar_vec
end

"""
    obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-6)

Perform dimensionality reduction using principal component analysis on
the variance y_var. Only eigenvectors with eigenvalues

    eigval >  allowed_var_loss * maximum(eigvals)

are retained.
Inputs:
 - y_mean :: Mean of the observations.
 - y_var :: Variance of the observations.
 - allowed_var_loss :: Lower limit for eigenvalues retained.
Outputs:
 - y_pca :: Projection of y_mean onto principal subspace spanned by eigenvectors.
 - y_var_pca :: Projection of y_var on principal subspace.
 - P_pca :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
"""
function obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-6;
    pool_norm = false, eigval_norm = false)
    eig = eigen(y_var)
    eigvals, eigvecs = eig; # eigvecs is matrix with eigvecs as cols
    # Get index of leading eigenvalues
    leading_eigs = findall(>(allowed_var_loss), eigvals/maximum(eigvals))
    P_pca = eigvecs[:, leading_eigs]
    λ_pca = eigvals[leading_eigs]
    # Check correct PCA projection
    @assert Diagonal(λ_pca) ≈ P_pca' * y_var * P_pca
    if pool_norm
        λ_pca = λ_pca/maximum(λ_pca)
    elseif eigval_norm
        λ_pca = ones(length((λ_pca)))
    end
    # Project mean
    y_pca = P_pca' * y_mean
    y_var_pca = Diagonal(λ_pca)
    return y_pca, y_var_pca, P_pca
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

"""
    normalize_profile(profile_vec, var_name, var_vec)

Perform normalization of profiles contained in profile_vec
using the standard deviation associated with each variable in
var_name. Variances for each variable are contained
in var_vec.
"""
function normalize_profile(profile_vec, var_name, var_vec)
    prof_vec = deepcopy(profile_vec)
    dim_variable = Integer(length(profile_vec)/length(var_name))
    for i in 1:length(var_name)
        prof_vec[dim_variable*(i-1)+1:dim_variable*i] =
            prof_vec[dim_variable*(i-1)+1:dim_variable*i] ./ sqrt(var_vec[i])
    end
    return prof_vec
end

function get_timevar_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=0.0,
                     getFullHeights=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                     normalize=false)

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
            prof_vec_itp = interpolate( (z_les, 1:tf_index-ti_index+1),
                prof_vec[1 + length(z_les)*(i-1) : i*length(z_les), :],
                ( Gridded(Linear()), NoInterp() ))
            prof_vec_zscm = cat(prof_vec_zscm,
                prof_vec_itp(z_scm, 1:tf_index-ti_index+1), dims=1)
            maxvar_vec[i] = maximum(var(prof_vec_zscm[1 + length(z_scm)*(i-1) : i*length(z_scm), :], dims=2))
        end

        cov_mat = cov(prof_vec_zscm, dims=2)
        if normalize
            for i in 1:num_outputs
                cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), :] = (
                   cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), :] ./ sqrt(maxvar_vec[i]) )
                cov_mat[:, 1 + length(z_scm)*(i-1) : i*length(z_scm)] = (
                   cov_mat[:, 1 + length(z_scm)*(i-1) : i*length(z_scm)] ./ sqrt(maxvar_vec[i]) )
            end
        end
    else
        cov_mat = cov(prof_vec, dims=2)
    end
    return cov_mat, maxvar_vec
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
    # Wrapper around SCAMPy in original output coordinates
    g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names, y_names, scm_dir, ti, tf)

    scm_dir = "/home/ilopezgo/SCAMPy/"
    params_cons_i = deepcopy(transform_unconstrained_to_constrained(priors, params))    
    params_cons_i = [row[:] for row in eachrow(params_cons_i')]
    N_ens = size(params_cons_i, 1)
    g_ens_arr = pmap(g_, params_cons_i) # [N_ens N_output]
    @assert size(g_ens_arr, 1) == N_ens
    N_out = size(g_ens_arr, 2)
    # If more than 1/4 of outputs are over limit lim, deemed as unstable simulation
    uns_vals_frac = sum(count.(x->x>lim, g_ens_arr), dims=2)./N_out
    unstable_point_inds = findall(x->x>0.25, uns_vals_frac)
    println(string("Unstable parameter indices: ", unstable_point_inds))
    # Recursively eliminate all unstable parameters
    if !isempty(unstable_point_inds)
        println(length(unstable_point_inds), " unstable parameters found:" )
        for j in length(unstable_point_inds)
            println(params[:, unstable_point_inds[j]])
        end
        println("Sampling new parameters from prior...")
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

function compute_errors(g_arr, y)
    diffs = [g - y for g in g_arr]
    errors = map(x->dot(x,x), diffs)
    return errors
end
