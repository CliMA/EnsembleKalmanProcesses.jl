# # [Learning the Pearmibility field in a Darcy flow from noisy sparse observations] 

# In this example we hope to illustrate function learning. One may wish to use function learning in cases where the underlying parameter of interest is actual a finite-dimensional approximation (e.g. spatial discretization) of some "true" function. Treating such an object directly will lead to increasingly high-dimensional learning problems as the spatial resolution is increased, resulting in poor computational scaling and increasingly ill-posed inverse problems. Treating the object as a discretized function from a function space, one can learn coefficients not in the standard basis, but instead in a basis of this function space, it is commonly the case that functions will have relatively low effective dimension, and will be depend only on the spatial discretization due to discretization error, that should vanish as resolution is increased. 

# We will solve for an unknown permeability field ``\kappa`` governing the pressure field of a Darcy flow on a square 2D domain. To learn about the permeability we shall take few pointwise measurements of the solved pressure field within the domain. The forward solver is a simple finite difference scheme taken and modified from code [here](https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl). 

# First we load standard packages
using LinearAlgebra
using Distributions
using Random
using JLD2

# the package to define the function distributions
import GaussianRandomFields # we wrap this so we don't want to use "using"
const GRF = GaussianRandomFields

# and finally the EKP packages
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

# We include the forward solver here
include("GModel.jl")

# Then link some outputs for figures and plotting
fig_save_directory = joinpath(@__DIR__, "output")
data_save_directory = joinpath(@__DIR__, "output")
if !isdir(fig_save_directory)
    mkdir(fig_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end# TOML interface for fitting parameters of a sinusoid

PLOT_FLAG = true
if PLOT_FLAG
    using Plots
    @info "Plotting enabled, this will reduce code performance. Figures stored in $fig_save_directory"
end

# Set a random seed.
seed = 100234
rng = Random.MersenneTwister(seed)

# Define the spatial domain and discretization 
dim = 2
N, L = 120, 1.0
pts_per_dim = LinRange(0, L, N)
obs_ΔN = 10

# To provide a simple test case, we assume that the true function parameter is a particular sample from the function space we set up to define our prior. More precisely we choose a value of the truth that doesnt have a vanishingly small probability under the prior defined by a probability distribution over functions; here taken as a family of Gaussian Random Fields (GRF). The function distribution is characterized by a covariance function - here a Matern kernel which assumes a level of smoothness over the samples from the distribution. We define an appropriate expansion of this distribution, here based on the Karhunen-Loeve expansion (similar to an eigenvalue-eigenfunction expansion) that is truncated to a finite number of terms, known as the degrees of freedom (`dofs`). The `dofs` define the effective dimension of the learning problem, decoupled from the spatial discretization. Explicitly, larger `dofs` may be required to represent multiscale functions, but come at an increased dimension of the parameter space and therefore a typical increase in cost and difficulty of the learning problem.

smoothness = 1.5
corr_length = 0.25
dofs = 30

grf = GRF.GaussianRandomField(
    GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
    GRF.KarhunenLoeve(dofs),
    pts_per_dim,
    pts_per_dim,
)

# We define a wrapper around the GRF, and as the permeability field must be positive we introduce a domain constraint into the function distribution. Henceforth, the GRF is interfaced in the same manner as any other parameter distribution with regards to interface.
pkg = GRFJL()
distribution = GaussianRandomFieldInterface(grf, pkg) # our wrapper from EKP
domain_constraint = bounded_below(0) # make κ positive
pd = ParameterDistribution(Dict("distribution" => distribution, "name" => "kappa", "constraint" => domain_constraint)) # the fully constrained parameter distribution

# Now we have a function distribution, we sample a reasonably high-probability value from this distribution as a true value (here all degrees of freedom set with `u_{\mathrm{true}} = -0.5`). We use the EKP transform function to build the corresponding instance of the ``\kappa_{\mathrm{true}}``.
u_true = -0.3 * ones(dofs, 1) # the truth parameter
println("True coefficients: ")
println(u_true)
κ_true = transform_unconstrained_to_constrained(pd, u_true) # builds and constrains the function.  
κ_true = reshape(κ_true, N, N)

# Now we generate the data sample for the truth in a perfect model setting by evaluating the the model here, and observing it by subsampling in each dimension every `obs_ΔN` points, and add some observational noise
darcy = Setup_Param(pts_per_dim, obs_ΔN, κ_true)
println(" Number of observation points: $(darcy.N_y)")
pressure_2d_true = solve_Darcy_2D(darcy, κ_true)
y_noiseless = compute_obs(darcy, pressure_2d_true)

obs_pts_per_dim = (obs_ΔN:obs_ΔN:(length(pts_per_dim) - obs_ΔN)) ./ length(pts_per_dim)
N_obs = length(obs_pts_per_dim)


# flat noise
# obs_noise_cov = sqrt(0.05)^2 * (maximum(y_noiseless) - minimum(y_noiseless))^2 * I(length(y_noiseless))

# do a fun noise which is small in the center and large further out
flattened_well = [0.01 + 0.1*((i-0.5)^2+(j-0.5)^2) for i in obs_pts_per_dim for j in obs_pts_per_dim] # 0.01 at (0.5,0.5) and larger away from this (1 at corners)
obs_noise_cov = (maximum(y_noiseless) - minimum(y_noiseless))^2 * (sqrt(0.05)^2 * I(length(y_noiseless)) + Diagonal(flattened_well.^2))

truth_sample = vec(y_noiseless + rand(rng, MvNormal(zeros(length(y_noiseless)), obs_noise_cov)))


## Now we consider the design object, which is also a GRF in the observation domain, 
##
smoothness_des = 1.0
corr_length_des = 0.1
dofs_des = 10


grf_des = GRF.GaussianRandomField(
    GRF.CovarianceFunction(dim, GRF.Matern(smoothness_des, corr_length_des)),
    GRF.KarhunenLoeve(dofs_des),
    obs_pts_per_dim,
    obs_pts_per_dim,
)

pkg_des = GRFJL()
distribution_des = GaussianRandomFieldInterface(grf_des, pkg_des) # our wrapper from EKP
domain_constraint_des = bounded_below(0) # make κ positive
prior_des = ParameterDistribution(Dict("distribution" => distribution_des, "name" => "design", "constraint" => domain_constraint_des)) # the fully constrained parameter distribution


# we then create a selection of different designs
N_iter_des = 5
#N_des = 20 # >= 3 for plotting
#initial_des = construct_initial_ensemble(rng, prior_des, N_des)
zero_data = [1.49, 15.6, 0.0355] #[0.0, 0.0, 0.0] # minimize [u-u_true, 1/Autil, 1/logDutil] 
noise_about_zero = Diagonal([0.01, 0.2, 1e-5])
ekp_des = EKP.EnsembleKalmanProcess(zero_data, noise_about_zero, Unscented(prior_des;impose_prior=true ), verbose=true, scheduler = EKP.DataMisfitController(terminate_at=1000.0))
N_des = get_N_ens(ekp_des)
# inner loop storage
u_final_means = []
u_final_covs = []
g_final_means = []
κ_final_means = []
κ_final_vars = []

#outer loop storage
u_final_des_means= []
u_final_des_covs= []
g_final_des_means= []
des_final_means= []
des_final_vars = []

err_des = []
final_it_des= [N_iter_des]

n_levels = 20 # for plots

#for inner-loop EKS
N_ens = 30 # number of ensemble members
N_iter = 4 # number of EKS iterations
        
for iter in 1:N_iter_des # outer loop optimization

    design_cols = get_ϕ_final(prior_des, ekp_des)
    designs = []
    designs_square = []
    for col in eachcol(design_cols)
        design = col ./ (sum(col)/(length(col)))
        design_square = reshape(col, N_obs , N_obs)
        push!(designs, design)
        push!(designs_square, design_square) # for plotting
    end
    
    
    if PLOT_FLAG && iter ==1 
        
        # create a sample input
        κ_col = transform_unconstrained_to_constrained(pd, sample(pd,1))
        κ = reshape(κ_col, N, N)
        
        # run darcy
        pressure_2d = solve_Darcy_2D(darcy, κ)
        model_out = compute_obs(darcy, pressure_2d)
        
        # look at effect on observing darcy with 3 of the design sets
        obs_diff = y_noiseless - model_out
        weighted_obs_diffs = [reshape(des .* obs_diff, N_obs, N_obs)  for des in designs]  
        
        
        gr(size =(1500,1200),legend = false)
        p1 = contour(pts_per_dim, pts_per_dim, κ_true, fill = true, levels = n_levels, title="true param", colorbar = true)
        p2 = contour(pts_per_dim, pts_per_dim, pressure_2d_true, fill = true, title="global data", levels = n_levels, colorbar = true)
        p3 = contour(pts_per_dim, pts_per_dim, pressure_2d, fill = true, title="sample model output", levels = n_levels, colorbar = true)
        p4 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[1], fill = true, title="design 1", levels = n_levels, colorbar = true)
        p5 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[2], fill = true, title="design 2",levels = n_levels, colorbar = true)
        p6 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[3], fill = true, title="design 3",levels = n_levels, colorbar = true)
        p7 = contour(obs_pts_per_dim, obs_pts_per_dim, weighted_obs_diffs[1], title="weighted diff 1", fill = true, levels = n_levels, colorbar = true)
        p8 = contour(obs_pts_per_dim, obs_pts_per_dim, weighted_obs_diffs[2], title="weighted diff 2", fill = true, levels = n_levels, colorbar = true)
        p9 = contour(obs_pts_per_dim, obs_pts_per_dim, weighted_obs_diffs[3], title="weighted diff 3", fill = true, levels = n_levels, colorbar = true)
        l = @layout [a b c; d e f; g h i]
        plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout = l)
        savefig(plt, joinpath(fig_save_directory,"data_and_sample_3_designs.png"))
    end
    
    # Now we set up the Bayesian inversion algorithm. The prior we have already defined to construct our truth
    prior = pd
    
    # We define some algorithm parameters, here we take ensemble members larger than the dimension of the parameter space
  
    
    # now solve some problems with these observations (no emulators) to illustrate the resulting posteriors (use EKS for all)
    

    u_final_means_tmp = []
    u_final_covs_tmp = []
    g_final_means_tmp = []
    κ_final_means_tmp = []
    κ_final_vars_tmp = []
    
    for (des_idx, design) in enumerate(designs) # inner loop optimizations
        
        truth_at_design = truth_sample .* design
        
        # as obs noise is just a percent of the data range^2, then we should be able to scale it with the design^2 do it in a flat way
        noise_des_scaling = 1 # use the mean of the design (1)
        #    noise_des_scaling = maximum(design)^2 # 
        obs_noise_cov_at_design = Diagonal(obs_noise_cov * noise_des_scaling)
        
        # We sample the initial ensemble from the prior, and create the EKP object as an EKI algorithm using the `Inversion()` keyword
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        ekp = EKP.EnsembleKalmanProcess(initial_params, truth_at_design, obs_noise_cov_at_design, Sampler(prior))
        #ekp = EKP.EnsembleKalmanProcess(truth_sample, obs_noise_cov, Unscented(prior))
        
        @info "Begin sampling with design $(des_idx)"
        err = []
        final_it = [N_iter]
        for i in 1:N_iter
            params_i = get_ϕ_final(prior, ekp)
            g_ens = reshape(design,:,1) .* run_G_ensemble(darcy, params_i) # mult cols of g_ens by the design
            terminate = EKP.update_ensemble!(ekp, g_ens)
            push!(err, get_error(ekp)[end]) #mean((params_true - mean(params_i,dims=2)).^2)
            println("Iteration: " * string(i) * ", Error: " * string(err[i]))
            if !isnothing(terminate)
            final_it[1] = i - 1
                break
            end
        end
        n_iter = final_it[1]
        @info "Final EKS time: $(sum(get_Δt(ekp)))"
        # We plot first the prior ensemble mean and pointwise variance of the permeability field, and also the pressure field solved with the ensemble mean. Each ensemble member is stored as a column and therefore for uses such as plotting one needs to reshape to the desired dimension.
        
        push!(u_final_means_tmp, mean(get_u_final(ekp), dims=2))
        push!(u_final_covs_tmp, cov(get_u_final(ekp), dims=2))
        push!(g_final_means_tmp, reshape(mean(get_g_final(ekp), dims=2), N_obs, N_obs))
        push!(κ_final_means_tmp, reshape(mean(get_ϕ_final(prior, ekp), dims=2),N, N))
        push!(κ_final_vars_tmp, reshape(var(get_ϕ_final(prior, ekp), dims=2), N, N))
        
        if PLOT_FLAG && (des_idx == 1 && iter ==1)
            gr(size = (1500, 400), legend = false)
            prior_κ_ens = get_ϕ(prior, ekp, 1)
            κ_ens_mean = reshape(mean(prior_κ_ens, dims = 2), N, N)
            p1 = contour(pts_per_dim, pts_per_dim, κ_ens_mean', fill = true, levels = n_levels, title = "kappa mean", colorbar = true)
            κ_ens_ptw_var = reshape(var(prior_κ_ens, dims = 2), N, N)
            p2 = contour(
                pts_per_dim,
                pts_per_dim,
                κ_ens_ptw_var',
            fill = true,
                levels = n_levels,
                title = "kappa var",
                colorbar = true,
            )
            pressure_2d = solve_Darcy_2D(darcy, κ_ens_mean)
            p3 = contour(pts_per_dim, pts_per_dim, pressure_2d', fill = true, levels = n_levels, title = "pressure", colorbar = true)
            l = @layout [a b c]
            plt = plot(p1, p2, p3, layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_prior.png")) # pre update
            
        end
        println("Final coefficients (ensemble mean):")
        println(get_u_mean_final(ekp))
        
        # We can compare this with the true permeability and pressure field: 
        if PLOT_FLAG && (des_idx == 1 && iter == 1)
            gr(size = (1000, 400), legend = false)
            p1 = contour(pts_per_dim, pts_per_dim, κ_true', fill = true, levels = n_levels, title = "kappa true", colorbar = true)
            p2 = contour(
                pts_per_dim,
                pts_per_dim,
                pressure_2d_true',
                fill = true,
                levels = n_levels,
                title = "pressure true",
                colorbar = true,
            )
            l = @layout [a b]
            plt = plot(p1, p2, layout = l)
            savefig(plt, joinpath(fig_save_directory, "output_true.png"))
        end
        
        # Finally the data is saved
        u_stored = get_u(ekp, return_array = false)
        g_stored = get_g(ekp, return_array = false)
        @save joinpath(data_save_directory, "parameter_storage_$(des_idx).jld2") u_stored
        @save joinpath(data_save_directory, "data_storage_$(des_idx).jld2") g_stored
    end

    
    push!(u_final_means, u_final_means_tmp)
    push!(u_final_covs, u_final_covs_tmp)
    push!(g_final_means, g_final_means_tmp)
    push!(κ_final_means, κ_final_means_tmp)
    push!(κ_final_vars, κ_final_vars_tmp)
    
    if PLOT_FLAG
        
        gr(size =(1500,1600),legend = false)
        
        p1 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[1], fill = true, title="design 1", levels = n_levels, colorbar = true)
        p2 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[2], fill = true, title="design 2",levels = n_levels, colorbar = true)
        p3 = contour(obs_pts_per_dim, obs_pts_per_dim, designs_square[3], fill = true, title="design 3",levels = n_levels, colorbar = true)
        p4 = contour(pts_per_dim, pts_per_dim, κ_final_means_tmp[1], fill = true, title="final mean 1", levels = n_levels, colorbar = true)
        p5 = contour(pts_per_dim, pts_per_dim, κ_final_means_tmp[2], fill = true, title="final mean 2",levels = n_levels, colorbar = true)
        p6 = contour(pts_per_dim, pts_per_dim, κ_final_means_tmp[3], fill = true, title="final mean 3",levels = n_levels, colorbar = true)
        p7 = contour(pts_per_dim, pts_per_dim, κ_final_vars_tmp[1], title="final var 1", fill = true, levels = n_levels, colorbar = true)
        p8 = contour(pts_per_dim, pts_per_dim, κ_final_vars_tmp[2], title="final var 2", fill = true, levels = n_levels, colorbar = true)
        p9 = contour(pts_per_dim, pts_per_dim, κ_final_vars_tmp[3], title="final var 3", fill = true, levels = n_levels, colorbar = true)
        p10 = contour(obs_pts_per_dim, obs_pts_per_dim, g_final_means_tmp[1], title="final output 1", fill = true, levels = n_levels, colorbar = true)
        p11 = contour(obs_pts_per_dim, obs_pts_per_dim, g_final_means_tmp[2], title="final output 2", fill = true, levels = n_levels, colorbar = true)
        p12 = contour(obs_pts_per_dim, obs_pts_per_dim, g_final_means_tmp[3], title="final output 3", fill = true, levels = n_levels, colorbar = true)
        
        l = @layout [a b c; d e f; g h i; j k l]
        plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, layout = l)
        savefig(plt, joinpath(fig_save_directory, "posterior_at_3_designs_$(iter).png")) # pre update
        
    end
    
    # calculate the evals (safer than taking determinants straight out)
    u_cov_evs = eigvals.(u_final_covs_tmp)
    Autil = [1 ./ sum(c) for c in u_cov_evs] # could also div this by (dofs-1)
    magnitude_range = 1e-2
    
    # compute D-util with only the evals within 1000 of largest
    D_tols = maximum.(u_cov_evs) .* magnitude_range
    safe_prod = [prod(ev[ev .> tol]) for (tol, ev) in zip(D_tols,u_cov_evs)]
    logDutil = log.(1 ./ safe_prod)
    @info """
    
Bias from each design
Designs: $(collect(1:N_des))
Bias in u: $([norm(m - u_true) for m in u_final_means_tmp])
Bias in κ: $([norm(m - κ_true) for m in κ_final_means_tmp])
Utilities for each design
*******************************
A-Utility: $(Autil)
safe log-D-Utility (computed with top $(100*(1.0 - magnitude_range))% of evals): $(logDutil)
     
    """
    # Now we have the utility, we update the designs:
    g_ens = zeros(3,N_des)
    g_ens[1,:] = [norm(m - u_true) for m in u_final_means[end]] # compare with 0
    g_ens[2,:] = 1 ./ Autil # compare with 0
    g_ens[3,:] = 1 ./ logDutil # compare with 0
    terminate = EKP.update_ensemble!(ekp_des, g_ens)
    push!(err_des, get_error(ekp_des)[end]) #mean((params_true - mean(params_i,dims=2)).^2)
    println("Design Iteration: " * string(iter) * ", Error: " * string(err_des[iter]))
    if !isnothing(terminate)
        final_it_des[1] = iter - 1
        break
    end

    push!(u_final_des_means, mean(get_u_final(ekp_des), dims=2))
    push!(u_final_des_covs, cov(get_u_final(ekp_des), dims=2))
    push!(g_final_des_means, mean(get_g_final(ekp_des))) # scalar
    push!(des_final_means, reshape(mean(get_ϕ_final(prior_des, ekp_des), dims=2), N_obs, N_obs))
    push!(des_final_vars, reshape(var(get_ϕ_final(prior_des, ekp_des), dims=2), N_obs, N_obs))
    
end


# now we look at the final design
u_cov_evs = eigvals.(u_final_covs[end])
Autil = [1 ./ sum(c) for c in u_cov_evs]

# compute D-util with only the evals within 1000 of largest
magnitude_range = 1e-2
D_tols = maximum.(u_cov_evs) .* magnitude_range
safe_prod = [prod(ev[ev .> tol]) for (tol, ev) in zip(D_tols,u_cov_evs)]
logDutil = log.(1 ./ safe_prod)

@info """
    
Bias from final designs
Designs: $(collect(1:N_des))
Bias in u: $([norm(m - u_true) for m in u_final_means[end]])
Bias in κ: $([norm(m - κ_true) for m in κ_final_means[end]])
Utilities for final designs
*******************************
A-Utility: $(Autil)
     
    """
