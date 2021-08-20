using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob
using JSON
using Random
# EKP modules
using EnsembleKalmanProcesses.ParameterDistributionStorage
# TurbulenceConvection.jl
using TurbulenceConvection
tc_dir = dirname(dirname(pathof(TurbulenceConvection)));
include(joinpath(tc_dir, "integration_tests", "utils", "main.jl"))


Base.@kwdef struct ReferenceModel
    "Vector of reference variable names"
    y_names::Vector{String}

    "Root directory for reference LES data"
    les_root::String
    "Name of LES reference simulation file"
    les_name::String
    "Suffix of LES reference simulation file"
    les_suffix::String

    "Root directory for SCM data (used for interpolation)"
    scm_root::String
    "Name of SCM reference simulation file"
    scm_name::String
    "Suffix of SCM reference simulation file"
    scm_suffix::String = "00000"

    # TODO: Make t_start and t_end vectors for multiple time intervals per reference model.
    "Start time for computing statistics over"
    t_start::Real
    "End time for computing statistics over"
    t_end::Real
end

les_dir(m::ReferenceModel) = data_directory(m.les_root, m.les_name, m.les_suffix)
scm_dir(m::ReferenceModel) = data_directory(m.scm_root, m.scm_name, m.scm_suffix)
data_directory(root::S, name::S, suffix::S) where S<:AbstractString = joinpath(root, "Output.$name.$suffix")

namelist_directory(root::String, m::ReferenceModel) = namelist_directory(root, m.scm_name)
namelist_directory(root::S, casename::S) where S<:AbstractString = joinpath(root, "namelist_$casename.in")

num_vars(m::ReferenceModel) = length(m.y_names)


Base.@kwdef struct ReferenceStatistics{FT <: Real}
    "Reference data, length: nSim * n_vars * n_zLevels(possibly reduced by PCA)"
    y::Vector{FT} # yt
    "Data covariance matrix, dims: (y,y) (possibly reduced by PCA)"
    Γ::Array{FT, 2}  # Γy
    "Vector (length: nSim) of normalizing factors (length: n_vars)"
    norm_vec::Vector{Array{FT, 1}}  # pool_var_list

    "Vector (length: nSim) of PCA projection matrices with leading eigenvectors as columns"
    pca_vec::Vector{Union{Array{FT, 2}, UniformScaling}}  # P_pca_list

    "Full reference data vector, length: nSim * n_vars * n_zLevels"
    y_full::Vector{FT}  # yt_big
    "Full covariance matrix, dims: (y,y)"
    Γ_full::Array{FT, 2}  # yt_var_big

    function ReferenceStatistics(RM::Vector{ReferenceModel}, model_type::Symbol, perform_PCA::Bool, normalize::Bool, FT=Float64)
        # Init arrays
        y = FT[]  # yt
        Γ_vec = Array{FT, 2}[]  # yt_var_list
        y_full = FT[]  # yt_big
        Γ_full_vec = Array{FT, 2}[]  # yt_var_list_big
        pca_vec = []  # P_pca_list
        norm_vec = Vector[]  # pool_var_list

        for m in RM
            # Get (interpolated and pool-normalized) observations, get pool variance vector
            y_, y_var_, pool_var = get_obs(model_type,
                m, z_scm = get_profile(scm_dir(m), ["z_half"]), normalize,
            )

            push!(norm_vec, pool_var)
            if perform_PCA
                y_pca, y_var_pca, P_pca = obs_PCA(y_, y_var_)
                append!(y, y_pca)
                push!(Γ_vec, y_var_pca)
                push!(pca_vec, P_pca)
            else
                append!(y, y_)
                push!(Γ_vec, y_var_)
                push!(pca_vec, 1.0I)
            end
            # Save full dimensionality (normalized) output for error computation
            append!(y_full, y_)
            push!(Γ_full_vec, y_var_)
        end
        # Construct global observational covariance matrix, TSVD
        indep_noise = 1e-3I
        Γ = cat(Γ_vec..., dims=(1,2)) + indep_noise
        @assert isposdef(Γ)
    
        Γ_full = cat(Γ_full_vec..., dims=(1,2)) + indep_noise
        return new{FT}(y, Γ, norm_vec, pca_vec, y_full, Γ_full)
    end
end

pca_length(RS::ReferenceStatistics) = length(RS.y)
full_length(RS::ReferenceStatistics) = length(RS.y_full)


"""
    run_SCM(
        u::Vector{FT},
        u_names::Vector{String},
        RM::Vector{ReferenceModel},
        RS::ReferenceStatistics,
    ) where FT<:Real

Run the single-column model (SCM) using a set of parameters u 
and return the value of outputs defined in y_names, possibly 
after normalization and projection onto lower dimensional 
space using PCA.

Inputs:
 - u                :: Values of parameters to be used in simulations.
 - u_names          :: SCAMPy names for parameters `u`.
 - RM               :: Vector of `ReferenceModel`s
 - RS               :: reference statistics for simulation
Outputs:
 - sim_dirs         :: Vector of simulation output directories
 - g_scm            :: Vector of model evaluations concatenated for all flow configurations.
 - g_scm_pca        :: Projection of `g_scm` onto principal subspace spanned by eigenvectors.
"""
function run_SCM(
    u::Vector{FT},
    u_names::Vector{String},
    RM::Vector{ReferenceModel},
    RS::ReferenceStatistics,
) where FT<:Real

    g_scm = zeros(0)
    g_scm_pca = zeros(0)
    sim_dirs = String[]

    for (i, m) in enumerate(RM)
        # create temporary directory to store SCAMPy data in
        tmpdir = mktempdir(pwd())

        # run TurbulenceConvection.jl. Get output directory for simulation data
        sim_dir = run_SCM_handler(m, tmpdir, u, u_names)
        push!(sim_dirs, sim_dir)

        g_scm_flow = get_profile(m, sim_dir)
        # normalize
        g_scm_flow = normalize_profile(g_scm_flow, length(m.y_names), RS.norm_vec[i])
        append!(g_scm, g_scm_flow)

        # perform PCA reduction
        append!(g_scm_pca, RS.pca_vec[i]' * g_scm_flow)
    end

    # penalize nan-values in output
    any(isnan.(g_scm)) && warn("NaN-values in output data")
    g_scm[isnan.(g_scm)] .= 1e5

    g_scm_pca[isnan.(g_scm_pca)] .= 1e5
    println("LENGTH OF G_SCM_ARR : ", length(g_scm))
    println("LENGTH OF G_SCM_ARR_PCA : ", length(g_scm_pca))
    return sim_dirs, g_scm, g_scm_pca
end


"""
    run_SCM_handler(
        m::ReferenceModel,
        tmpdir::String,
        u::Array{FT, 1},
        u_names::Array{String, 1},
    ) where {FT<:AbstractFloat}

Run a list of cases using a set of parameters `u_names` with values `u`,
and return a list of directories pointing to where data is stored for 
each simulation run.

Inputs:
 - m            :: Reference model
 - tmpdir       :: Directory to store simulation results in
 - u            :: Values of parameters to be used in simulations.
 - u_names      :: SCAMPy names for parameters `u`.
Outputs:
 - output_dirs  :: list of directories containing output data from the SCAMPy runs.
"""
function run_SCM_handler(
    m::ReferenceModel,
    tmpdir::String,
    u::Array{FT, 1},
    u_names::Array{String, 1},
) where {FT<:AbstractFloat}

    # fetch default namelist
    inputdir = scm_dir(m)
    namelist = JSON.parsefile(namelist_directory(inputdir, m))

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
    namelist_path = namelist_directory(tmpdir, m)
    open(namelist_path, "w") do io
        JSON.print(io, namelist, 4)
    end

    # run TurbulenceConvection.jl with modified parameters
    main(namelist)
    
    return data_directory(tmpdir, m.scm_name, uuid)
end


"""
    get_obs(
        obs_type::Symbol,
        m::ReferenceModel;
        z_scm::Union{Vector{FT}, Nothing} = nothing,
    ) where FT<:Real

Get observations for variables y_names, interpolated to
z_scm (if given), and possibly normalized with respect to the pooled variance.

Inputs:
 - obs_type     :: Either :les or :scm
 - m            :: Reference model
 - z_scm :: If given, interpolate LES observations to given levels.
Outputs:
 - y_ :: Mean of observations, possibly interpolated to z_scm levels.
 - y_tvar :: Observational covariance matrix, possibly pool-normalized.
 - pool_var :: Vector of vertically averaged time-variance, one entry for each variable
"""
function get_obs(
    obs_type::Symbol,
    m::ReferenceModel,
    normalize::Bool;
    z_scm::Union{Vector{FT}, Nothing} = nothing,
) where FT<:Real
    les_names = get_les_names(m.y_names, les_dir(m))
    
    # True observables from SCM or LES depending on `obs_type` flag
    y_names, sim_dir = if obs_type == :scm
        m.y_names, scm_dir(m)
    elseif obs_type == :les
        les_names, les_dir(m)
    else
        error("Unknown observation type $obs_type")
    end

    # For now, we always use LES to construct covariance matrix
    y_tvar, pool_var = get_time_covariance(
        m, les_dir(m), les_names, z_scm=z_scm,
    )

    norm_vec = if normalize
        pool_var
    else
        ones(size(pool_var))
    end

    # Get true observables
    y_highres = get_profile(m, sim_dir, y_names)
    # normalize
    y_highres = normalize_profile(y_highres, num_vars(m), norm_vec)

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
    return y_, y_tvar, norm_vec
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


function get_profile(m::ReferenceModel, sim_dir::String) 
    get_profile(m, sim_dir, m.y_names)
end


function get_profile(m::ReferenceModel, sim_dir::String, y_names::Vector{String})
    get_profile(sim_dir, y_names, ti=m.t_start, tf=m.t_end)
end


function get_profile(
    sim_dir::String,
    var_name::Vector{String};
    ti::Real = 0.0,
    tf = nothing
)
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
function normalize_profile(profile_vec, n_vars, var_vec)
    prof_vec = deepcopy(profile_vec)
    dim_variable = Integer(length(profile_vec)/n_vars)
    for i in 1:n_vars
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
function get_time_covariance(
    m::ReferenceModel,
    sim_dir::String,
    var_names::Vector{String};
    getFullHeights=false,
    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
)

    t = nc_fetch(sim_dir, "timeseries", "t")
    # Find closest interval in data
    ti_index = argmin( broadcast(abs, t.-m.t_start) )
    tf_index = argmin( broadcast(abs, t.-m.t_end) )
    ts_vec = zeros(0, length(ti_index:tf_index))
    num_outputs = length(var_names)
    pool_var = zeros(num_outputs)

    for i in 1:num_outputs
        var_ = nc_fetch(sim_dir, "profiles", var_names[i])
        # LES vertical fluxes are per volume, not mass
        if occursin("resolved_z_flux", var_names[i])
            rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
            var_ = var_.*rho_half
        end
        # Store pooled variance
        pool_var[i] = mean(var(var_[:, ti_index:tf_index], dims=2))  # vertically averaged time-variance of variable
        # normalize timeseries
        ts_var_i = var_[:, ti_index:tf_index]./ sqrt(pool_var[i])
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
    return cov_mat, pool_var
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
    compute_errors(g_arr, y)

Computes the L2-norm error of each elmt of g_arr
wrt vector y.
"""
function compute_errors(g_arr, y)
    diffs = [g - y for g in g_arr]
    errors = map(x->dot(x,x), diffs)
    return errors
end
