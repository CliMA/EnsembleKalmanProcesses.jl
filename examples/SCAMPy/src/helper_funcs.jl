using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JLD
using JSON
using Random
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
# TurbulenceConvection.jl
using TurbulenceConvection
tc_dir = dirname(dirname(pathof(TurbulenceConvection)));
include(joinpath(tc_dir, "integration_tests", "utils", "main.jl"))

"""
    run_SCM(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
        scm_data_root::String,
        scm_names::Array{String, 1},
        ti::Union{Array{FT,1}, Array{Array{FT,1},1}},
        tf::Union{Array{FT,1}, Array{Array{FT,1},1}, Nothing} = nothing;
        norm_var_list = nothing,
        P_pca_list = nothing,
    ) where {FT<:AbstractFloat}

Run the single-column model (SCM) using a set of parameters u 
and return the value of outputs defined in y_names, possibly 
after normalization and projection onto lower dimensional 
space using PCA.

Inputs:
 - u                :: Values of parameters to be used in simulations.
 - u_names          :: SCAMPy names for parameters `u`.
 - y_names          :: Name of outputs requested for each flow configuration.
 - scm_data_root    :: Path to input data for the SCM model.
 - scm_names        :: Names of SCAMPy cases
 - ti               :: Vector of starting times for observation intervals. 
                        If `tf=nothing`, snapshots at `ti` are returned.
 - tf               :: Vector of ending times for observation intervals.
 - norm_var_list    :: Pooled variance vectors. If given, use to normalize output.
 - P_pca_list       :: Vector of projection matrices `P_pca` for each flow configuration.
Outputs:
 - sim_dirs         :: Vector of simulation output directories
 - g_scm            :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca        :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
"""
function run_SCM(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        y_names::Union{Array{String, 1}, Array{Array{String,1},1}},
        scm_data_root::String,
        scm_names::Array{String, 1},
        ti::Union{Array{FT,1}, Array{Array{FT,1},1}},
        tf::Union{Array{FT,1}, Array{Array{FT,1},1}, Nothing} = nothing;
        norm_var_list = nothing,
        P_pca_list = nothing,
    ) where {FT<:AbstractFloat}

    # Check parameter dimensionality
    @assert length(u_names) == length(u)

    # run SCAMPy and get simulation dirs
    sim_dirs = run_SCM_handler(u, u_names, scm_names, scm_data_root)

    # Check consistent time interval dims
    @assert length(ti) == length(sim_dirs)

    g_scm = zeros(0)
    g_scm_pca = zeros(0)

    if typeof(ti) == Array{FT,1} # 1 interval per simulation
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
        end
    elseif typeof(ti) == Array{Array{FT,1},1} # multiple intervals per simulation
        config_num = 1
        for (i, sim_dir) in enumerate(sim_dirs)
            y_names_ = typeof(y_names)==Array{Array{String,1},1} ? y_names[i] : y_names
            for (j, ti_j) in enumerate(ti[i]) # Loop on time intervals per sim
                tf_j = !isnothing(tf) ? tf[i][j] : nothing
                g_scm_flow = get_profile(sim_dir, y_names_, ti = ti_j, tf = tf_j)
                if !isnothing(norm_var_list)
                    g_scm_flow = normalize_profile(g_scm_flow, y_names_, norm_var_list[config_num])
                end
                append!(g_scm, g_scm_flow)
                if !isnothing(P_pca_list)
                    append!(g_scm_pca, P_pca_list[config_num]' * g_scm_flow)
                end
                config_num += 1
            end
        end
    end
    # penalize nan-values in output
    for i in eachindex(g_scm)
        g_scm[i] = isnan(g_scm[i]) ? 1.0e5 : g_scm[i]
    end
    if !isnothing(P_pca_list)
        for i in eachindex(g_scm_pca)
            g_scm_pca[i] = isnan(g_scm_pca[i]) ? 1.0e5 : g_scm_pca[i]
        end
        println("LENGTH OF G_SCM_ARR : ", length(g_scm))
        println("LENGTH OF G_SCM_ARR_PCA : ", length(g_scm_pca))
        return sim_dirs, g_scm, g_scm_pca
    else
        return sim_dirs, g_scm, nothing
    end
end


"""
    function run_SCM_handler(
        u::Array{FT, 1},  
        u_names::Array{String, 1},
        scm_names::String,
        scm_data_root::String,
    ) where {FT<:AbstractFloat}

Run a list of cases using a set of parameters `u_names` with values `u`,
and return a list of directories pointing to where data is stored for 
each simulation run.

Inputs:
 - u :: Values of parameters to be used in simulations.
 - u_names :: SCAMPy names for parameters `u`.
 - scm_names :: Names of SCAMPy cases to run
 - scm_data_root :: Path to SCAMPy case data (<scm_data_root>/Output.<scm_name>.00000)
Outputs:
 - output_dirs :: list of directories containing output data from the SCAMPy runs.
"""
function run_SCM_handler(
        u::Array{FT, 1},
        u_names::Array{String, 1},
        scm_names::Array{String, 1},
        scm_data_root::String,
    ) where {FT<:AbstractFloat}
    # create temporary directory to store SCAMPy data in
    tmpdir = mktempdir(pwd())

    # output directories
    output_dirs = String[]

    for casename in scm_names
        # For each scm case, fetch namelist
        inputdir = joinpath(scm_data_root, "Output.$casename.00000")
        namelist = JSON.parsefile(joinpath(inputdir, "namelist_$casename.in"))

        # update parameter values
        for (pName, pVal) in zip(u_names, u)
            namelist["turbulence"]["EDMF_PrognosticTKE"][pName] = pVal
        end

        # set random uuid
        uuid = basename(tmpdir)
        namelist["meta"]["uuid"] = uuid
        # set output dir to `tmpdir`
        namelist["output"]["output_root"] = tmpdir
        # write updated namelist to `tmpdir`
        namelist_path = joinpath(tmpdir, "namelist_$casename.in")
        open(namelist_path, "w") do io
            JSON.print(io, namelist, 4)
        end

        # run TurbulenceConvection.jl with modified parameters
        main(namelist)

        push!(output_dirs, joinpath(tmpdir, "Output.$casename.$uuid"))
    end  # end `scm_names` loop
    return output_dirs
end

"""
    get_obs(y_names, sim_dir, ti, tf;
            z_scm = nothing, normalize = false)

Get observations for variables y_names, interpolated to
z_scm (if given), and possibly normalized with respect to the pooled variance.

Inputs:
 - y_names :: Name of outputs requested for each flow configuration.
 - ti :: Vector of starting times for observation intervals. If tf=nothing,
         snapshots at ti are returned.
 - tf :: Vector of ending times for observation intervals.
 - z_scm :: If given, interpolate LES observations to given levels.
 - normalize :: If true, normalize observations and cov matrix by pooled variances.
 - perfect_model :: If true, the variable names are the same for the observation
                    and the single column model.
Outputs:
 - y_ :: Mean of observations, possibly interpolated to z_scm levels.
 - y_tvar :: Observational covariance matrix, possibly pool-normalized.
"""
function get_obs(y_names::Array{String, 1},
                    sim_dir::String,
                    ti::FT,
                    tf::FT;
                    z_scm::Union{Array{FT, 1}, Nothing} = nothing,
                    normalize = true,
                    perfect_model = false,
                    ) where {FT<:AbstractFloat}
    
    if perfect_model
        y_names_les = deepcopy(y_names)
    else
        y_names_les = get_les_names(y_names, sim_dir)
    end
    y_tvar, poolvar_vec = get_time_covariance(sim_dir, y_names_les,
        ti = ti, tf = tf, z_scm=z_scm, normalize=normalize)
    y_highres = get_profile(sim_dir, y_names_les, ti = ti, tf = tf)
    if normalize
        y_highres = normalize_profile(y_highres, y_names, poolvar_vec)
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
    return y_, y_tvar, poolvar_vec
end

"""
    obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-1)

Perform dimensionality reduction using principal component analysis on
the variance y_var. Only eigenvectors with eigenvalues that contribute
to the leading 1-allowed_var_loss variance are retained.
Inputs:
 - y_mean :: Mean of the observations.
 - y_var :: Variance of the observations.
 - allowed_var_loss :: Maximum variance loss allowed.
Outputs:
 - y_pca :: Projection of y_mean onto principal subspace spanned by eigenvectors.
 - y_var_pca :: Projection of y_var on principal subspace.
 - P_pca :: Projection matrix onto principal subspace, with leading eigenvectors as columns.
"""
function obs_PCA(y_mean, y_var, allowed_var_loss = 1.0e-1)
    eig = eigen(y_var)
    eigvals, eigvecs = eig; # eigvecs is matrix with eigvecs as cols
    # Get index of leading eigenvalues, eigvals are ordered from low to high in julia
    # This expression recovers 1 extra eigenvalue compared to threshold
    leading_eigs = findall(<(1.0-allowed_var_loss), -cumsum(eigvals)/sum(eigvals).+1)
    P_pca = eigvecs[:, leading_eigs]
    λ_pca = eigvals[leading_eigs]
    # Check correct PCA projection
    @assert Diagonal(λ_pca) ≈ P_pca' * y_var * P_pca
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

"""
    get_time_covariance(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=0.0,
                     getFullHeights=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                     normalize=false)

Obtain the covariance matrix of a group of profiles, where the covariance
is obtained in time.
Inputs:
 - sim_dir :: Name of simulation directory.
 - var_name :: List of variable names to be included.
 - ti, tf :: Initial and final times defining averaging interval.
 - z_scm :: If given, interpolates covariance matrix to this locations.
 - normalize :: Boolean specifying variable normalization.
"""
function get_time_covariance(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf=0.0,
                     getFullHeights=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                     normalize=false)

    t = nc_fetch(sim_dir, "timeseries", "t")
    # Find closest interval in data
    ti_index = argmin( broadcast(abs, t.-ti) )
    tf_index = argmin( broadcast(abs, t.-tf) )
    ts_vec = zeros(0, length(ti_index:tf_index))
    num_outputs = length(var_name)
    poolvar_vec = zeros(num_outputs)

    for i in 1:num_outputs
        var_ = nc_fetch(sim_dir, "profiles", var_name[i])
        # LES vertical fluxes are per volume, not mass
        if occursin("resolved_z_flux", var_name[i])
            rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
            var_ = var_.*rho_half
        end
        # Store pooled variance
        poolvar_vec[i] = mean(var(var_[:, ti_index:tf_index], dims=2))  # vertically averaged time-variance of variable
        ts_var_i = normalize ? var_[:, ti_index:tf_index]./ sqrt(poolvar_vec[i]) : var_[:, ti_index:tf_index]
        # Interpolate in space
        if !isnothing(z_scm)
            z_les = getFullHeights ? get_profile(sim_dir, ["z"]) : get_profile(sim_dir, ["z_half"])
            # Create interpolant
            ts_var_i_itp = interpolate(
                (z_les, 1:tf_index-ti_index+1),
                ts_var_i,
                ( Gridded(Linear()), NoInterp() )
            )
            # Interpolate
            ts_var_i = ts_var_i_itp(z_scm, 1:tf_index-ti_index+1)
        end
        ts_vec = cat(ts_vec, ts_var_i, dims=1)  # dims: (Nz*num_outputs, Nt)
    end
    cov_mat = cov(ts_vec, dims=2)  # covariance, w/ samples across time dimension (t_inds).
    return cov_mat, poolvar_vec
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
    scm_data_root::String, scm_names::Array{String, 1},
    ti::Union{FT, Array{FT,1}};
    tf::Union{FT, Array{FT,1}, Nothing}=nothing, lim::FT=1.0e4,) where {IT<:Int, FT}

    # Check dimensionality
    @assert length(param_names) == size(params, 1)
    # Wrapper around SCAMPy in original output coordinates
    g_(x::Array{Float64,1}) = run_SCM(
        x, param_names, y_names, scm_data_root, scm_names, ti, tf,
    )

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
        for j in 1:length(unstable_point_inds)
            println(params[:, unstable_point_inds[j]])
        end
        println("Sampling new parameters from prior...")
        new_params = construct_initial_ensemble(priors, length(unstable_point_inds))
        precondition_ensemble!(new_params, priors, param_names,
            y_names, scm_data_root, scm_names, ti, tf=tf, lim=lim)
        params[:, unstable_point_inds] = new_params
    end
    println("\nPreconditioning finished.")
    return
end

"""
    compute_errors(g_arr, y)

Computes the L2-norm error of each elmt of g_arr
wrt vector y.
"""
function compute_errors(g_arr, y)
    diffs = [g - y for g in g_arr]
    errors = map(x->dot(x,x), diffs)
    return errors
end

"""
    cov_from_cov_list(cov_list::Array{Array{FT,2},1}; indices=nothing)

Returns a block-diagonal covariance matrix constructed from covariances
within cov_list given by the indices. If isempty(indices), use all 
covariances to construct block-diagonal matrix.
"""
function cov_from_cov_list(cov_list::Array{Array{FT,2},1};
         indices=[]) where {FT<:AbstractFloat}
    size_ = isempty(indices) ? sum([length(cov[1,:]) for cov in cov_list]) :
        sum([length(cov[1,:]) for (i, cov) in enumerate(cov_list) if i in indices])

    cov_ = zeros(size_, size_)
    vars_num = 1
    for (index, small_cov) in enumerate(cov_list)
        if index in indices
            vars = length(small_cov[1,:])
            cov_[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = small_cov
            vars_num = vars_num+vars
        end
    end
    return cov_
end

"""
    vec_from_vec_list(vec_list::Array{Array{FT,1},1}; indices=[], return_mapping=false)

Returns a vector constructed from vectors within vec_list given by the
indices. If isempty(indices), use all vectors to construct returned vector.
If return_mapping, function returns the positions of all the elements used
to construct the returned vector.
"""
function vec_from_vec_list(vec_list::Array{Array{FT,1},1};
         indices=[], return_mapping=false) where {FT<:AbstractFloat}
    vector_ = zeros(0)
    elmt_num = []
    chosen_elmt_num = []
    for (index, small_vec) in enumerate(vec_list)
        index < 2 ? append!(elmt_num, 1:length(small_vec)) :
                    append!(elmt_num, elmt_num[end]+1:elmt_num[end]+length(small_vec))
        if index in indices
            append!(vector_, small_vec)
            append!(chosen_elmt_num, elmt_num[end] - length(small_vec) + 1 : elmt_num[end])
        end
    end
    if return_mapping
        return vector_, chosen_elmt_num
    else
        return vector_
    end
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
