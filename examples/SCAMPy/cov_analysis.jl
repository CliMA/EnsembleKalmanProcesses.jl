# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage
include(joinpath(@__DIR__, "helper_funcs.jl"))
using NPZ

###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn
param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
n_param = length(param_names)

###
###  Retrieve true LES samples from PyCLES data
###

# This is the true value of the observables (e.g. LES ensemble mean for EDMF)
ti = [10800.0, 28800.0, 10800.0, 18000.0]
tf = [14400.0, 32400.0, 14400.0, 21600.0]
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex

# Get observations
sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]
for i in range(0,1,length=2)
    normalize = Bool(i)
    suffix = normalize ? "_normalized" : "_wscales" 

    yt = zeros(0)
    yt_var_list = []
    yt_pca_list = zeros(0)
    yt_pca_var_list = []
    P_pca_list = []

    les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[1],".may20")
    sim_dir = string("Output.", sim_names[1],".00000")
    z_scm = get_profile(sim_dir, ["z_half"])
    yt_, yt_var_ = obs_LES(y_names[1], les_dir, ti[1], tf[1], z_scm = z_scm, normalize=normalize)
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)
    npzwrite("dycoms_z.npy", z_scm)
    yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
    append!(yt_pca_list, yt_pca)
    push!(yt_pca_var_list, yt_var_pca)
    push!(P_pca_list, P_pca)
    @assert length(yt_pca) == length(yt_var_pca[1,:])

    les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[2],".iles128wCov")
    sim_dir = string("Output.", sim_names[2],".00000")
    z_scm = get_profile(sim_dir, ["z_half"])
    yt_, yt_var_ = obs_LES(y_names[2], les_dir, ti[2], tf[2], z_scm = z_scm, normalize=normalize)
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)
    npzwrite("gabls_z.npy", z_scm)
    yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
    append!(yt_pca_list, yt_pca)
    push!(yt_pca_var_list, yt_var_pca)
    push!(P_pca_list, P_pca)
    @assert length(yt_pca) == length(yt_var_pca[1,:])

    les_dir = string("/groups/esm/ilopezgo/Output.Soares.dry11")
    sim_dir = string("Output.", sim_names[3],".00000")
    z_scm = get_profile(sim_dir, ["z_half"])
    yt_, yt_var_ = obs_LES(y_names[3], les_dir, ti[3], tf[3], z_scm = z_scm, normalize=normalize)
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)
    npzwrite("nieuwstadt_z.npy", z_scm)
    yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
    append!(yt_pca_list, yt_pca)
    push!(yt_pca_var_list, yt_var_pca)
    push!(P_pca_list, P_pca)
    @assert length(yt_pca) == length(yt_var_pca[1,:])

    les_dir = string("/groups/esm/ilopezgo/Output.Bomex.may18")
    sim_dir = string("Output.", sim_names[4],".00000")
    z_scm = get_profile(sim_dir, ["z_half"])
    yt_, yt_var_ = obs_LES(y_names[4], les_dir, ti[4], tf[4], z_scm = z_scm, normalize=normalize)
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)
    npzwrite("bomex_z.npy", z_scm)
    yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
    append!(yt_pca_list, yt_pca)
    push!(yt_pca_var_list, yt_var_pca)
    push!(P_pca_list, P_pca)
    @assert length(yt_pca) == length(yt_var_pca[1,:])

    yt_var = zeros(length(yt), length(yt))
    vars_num = 1
    for sim_covmat in yt_var_list
        vars = length(sim_covmat[1,:])
        yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = sim_covmat
        vars_num = vars_num+vars
    end

    yt_var_pca = zeros(length(yt_pca_list), length(yt_pca_list))
    vars_num = 1
    for sim_covmat in yt_pca_var_list
        vars = length(sim_covmat[1,:])
        yt_var_pca[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = sim_covmat
        vars_num = vars_num+vars
    end

    # Write covariances to file
    npzwrite("obs_cov"*suffix*".npy", yt_var)
    npzwrite("obs_cov_pca"*suffix*".npy", yt_var_pca)
end



