
# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using CairoMakie, ColorSchemes
using Random
using JLD2

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers


const EKP = EnsembleKalmanProcesses

# includes
include("lorenz_model.jl") # Contains coupled Lorenz 96 and integrator
include("data_processing.jl") # Contains processing to produce data and forward map


rng_seed = 79453
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

# Output figure save directory
homedir = pwd()
println(homedir)
output_directory = homedir * "/output/"
if ~isdir(output_directory)
    mkdir(output_directory)
end

case = "test"


# A function to run the ensemble over ϕ_i, generating trajectories and data
function run_G_ensemble(state, lsettings, ϕ_i, window, data_dim)
    N_ens = size(ϕ_i, 2)
    data_sample = zeros(data_dim, N_ens)

    for j in 1:N_ens
        # ϕ_i is n_params x n_ens
        params_i = LParams(
            ϕ_i[1:3, j], # F
            ϕ_i[1, j], # G
            ϕ_i[2, j], # h
            ϕ_i[3, j], # c
            ϕ_i[4, j], # b
        )
        new_state, new_t = lorenz_solve(state, lsettings, params_i)
        data_sample[:, j] = process_trajectory_to_data(window, new_state, new_t)
    end
    return data_sample
end


function main()
    # Lorenz 96 multiscale system with 1-slow timescale per slow timescale
    # Fast: Atmosphere with timescale difference c, nonlinear strength b, coupling y_k 
    # dy_k/dt = -cby_{k+1}(y_{k+2}-y_{k-1}) - cy_k + (hc/b)x_k
    # Slow: Ocean is just damping with e.g. seasonal forcing F and coupling strength G
    # dx_k/dt =  - x_k + (Fm + Fa*cos(Ff*2πt)) - Gy_k
    # Inspired from the coupled L63 model
    # Zhang, Liu, Rosati & Delworth (2012)
    # https://doi.org/10.3402/tellusa.v64i0.10963

    ###
    ###  Define the (true) parameters
    ###

    # Time scaling
    time_scale_factor = 0.05
    year = 360.0 * time_scale_factor
    month = year / 12.0 * time_scale_factor

    # Fast 
    h_true = 10.0
    c_true = 10.0# timescale difference
    b_true = 10.0

    # Slow
    F_true = [8.0, 2.0, 1 / year] # (mean,amplitude,frequency) const for now (maybe add seasonal cycle)
    G_true = h_true * c_true / b_true # for consistency in coupling terms

    true_params = LParams(F_true, G_true, h_true, c_true, b_true)
    param_names = ["F", "G", "h", "c", "b"]

    ###
    ###  Define the parameter priors (needed later for inversion)
    ###

    prior_F = ParameterDistribution(
        Dict(
            "name" => "F",
            "distribution" => Parameterized(MvNormal([1.0, 0.0, -2.0], I)),
            "constraint" => repeat([bounded_below(0)], 3),
        ),
    ) # gives 3-D dist on the order of~ [12,1,0.08]                                
    prior_G = constrained_gaussian(param_names[2], 5.0, 4.0, 0, Inf)
    prior_h = constrained_gaussian(param_names[3], 5.0, 4.0, 0, Inf)
    prior_c = constrained_gaussian(param_names[4], 5.0, 4.0, 0, Inf)
    prior_b = constrained_gaussian(param_names[5], 5.0, 4.0, 0, Inf)
    priors = combine_distributions([prior_F, prior_G, prior_h, prior_c, prior_b])

    ###
    ### Configuration of the forward map
    ###

    ## State evolution
    N = 40 # state size
    dt = 0.005 # integration dt


    ## Spin up
    spinup = year
    initial = rand(Normal(0.0, 0.01), 2 * N) # IC for fast and slow

    lsettings_spinup = LSettings(N, dt, spinup, initial)

    ## Gather statistics
    # e.g. run for 2 years, using last year for data, (& batched monthly)
    t_solve = 2 * year
    window_start = t_solve - year
    window_end = t_solve
    subwindow_size = Int64(floor(month / dt))
    slide_or_batch = "batch" #sliding window or disjoint batches

    lsettings = LSettings(N, dt, t_solve, nothing)
    window = ProcessingWindow(window_start, window_end, subwindow_size, slide_or_batch)

    ###
    ### Spin up and plot
    ###

    spunup_state, t = lorenz_solve(lsettings_spinup, true_params)
    @info "spin-up complete"

    fig = Figure(size = (900, 450))

    afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
    aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

    ss_freq = length(t) / 2010.0
    ss_rate = Int64(ceil(ss_freq))
    tplot = t[(10 * ss_rate + 1):ss_rate:end]
    hm =
        heatmap!(afast, tplot, 1:N, spunup_state[(N + 1):(2 * N), (10 * ss_rate + 1):ss_rate:end]', colormap = :Oranges)
    Colorbar(fig[1, 1][1, 2], hm)
    hm = heatmap!(aslow, tplot, 1:N, spunup_state[1:N, (10 * ss_rate + 1):ss_rate:end]', colormap = :Blues)
    Colorbar(fig[2, 1][1, 2], hm)

    # save
    save(joinpath(output_directory, case * "_spinup_allstate.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_spinup_allstate.pdf"), fig, pt_per_unit = 3)


    fig = Figure(size = (900, 450))

    afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
    aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

    lines!(afast, tplot, spunup_state[N + 1, (10 * ss_rate + 1):ss_rate:end], color = :red)
    lines!(aslow, tplot, spunup_state[1, (10 * ss_rate + 1):ss_rate:end], color = :blue)

    # save
    save(joinpath(output_directory, case * "_spinup_state1.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_spinup_state1.pdf"), fig, pt_per_unit = 3)
    @info "plotted spin-up"

    ###
    ### Generate perfect-model data
    ###

    #estimate covariance
    n_sample_cov = 100

    new_state, new_t = lorenz_solve(spunup_state, lsettings, true_params)
    data_sample = process_trajectory_to_data(window, new_state, new_t)

    data_samples = zeros(size(data_sample, 1), n_sample_cov)
    data_samples[:, 1] = data_sample

    fig = Figure(size = (900, 450))
    afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
    aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

    for i in 2:n_sample_cov
        # extend trajectory from end of new_state
        new_state, new_t = lorenz_solve(new_state, lsettings, true_params) #integrate another window from last state
        # calculate data sample
        data_samples[:, i] = process_trajectory_to_data(window, new_state, new_t) # process the window 

        #plot trajectory
        ss_freq = length(new_t) / 2010.0
        ss_rate = Int64(ceil(ss_freq))
        tplot = new_t[Int64(floor(length(new_t) / 2)):ss_rate:end]

        lines!(afast, tplot, new_state[N + 1, Int64(floor(length(new_t) / 2)):ss_rate:end], color = :red, alpha = 0.1)
        lines!(
            aslow,
            tplot,
            new_state[1, Int64(floor(length(new_t) / 2)):ss_rate:end],
            color = :blue,
            alpha = 0.1,
            label = "$(n_sample_cov) samples",
        )
        axislegend(aslow, merge = true, unique = true)
    end
    # save
    save(joinpath(output_directory, case * "_datatrajectory_state1.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_datatrajectory_state1.pdf"), fig, pt_per_unit = 3)

    Γ = cov(data_samples, dims = 2) # estimate covariance from samples
    # add a little additive and multiplicative inflation
    Γ += 1e4 * eps() * I # 10^-12 just to make things nonzero
    #blocksize = Int64(size(Γ, 1) / 5) # known block structure
    #meanblocks = [mean([Γ[i, i] for i in ((j - 1) * blocksize + 1):(j * blocksize)]) for j in 1:5]
    #Γ += 1e-4* kron(Diagonal(meanblocks),I(blocksize)) # this will add scaled noise to the diagonal scaled by the block

    y_mean = mean(data_samples, dims = 2)
    y = data_samples[:, shuffle(rng, 1:n_sample_cov)[1]] # random data point as the data
    fig = Figure(size = (450, 450))
    aΓ = Axis(fig[1, 1][1, 1])
    adata = Axis(fig[2, 1][1, 1])

    heatmap!(aΓ, Γ)
    sqrt_inv_Γ = sqrt(inv(Γ))
    series!(adata, (sqrt_inv_Γ * data_samples)', solid_color = :black, label = "normalized data") #plots each row as new plot
    # save
    save(joinpath(output_directory, case * "_datasamples.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_datasamples.pdf"), fig, pt_per_unit = 3)

    @info "constructed and plotted perfect experiment data and noise"

    # may need to condition Gamma for posdef etc. or add shrinkage estimation

    ###
    ### (a) Configure and solve ensemble inversion
    ###

    # EKP parameters
    N_ens = 30 # number of ensemble members
    N_iter = 10 # number of EKI iterations
    # initial parameters: N_params x N_ens
    initial_params = construct_initial_ensemble(rng, priors, N_ens)

    ekiobj = EKP.EnsembleKalmanProcess(
        initial_params,
        y,
        Γ,
        Inversion(),
        localization_method = Localizers.NoLocalization(),
        scheduler = DataMisfitController(terminate_at = 1e4),
        failure_handler_method = SampleSuccGauss(),
        verbose = true,
    )
    @info "Built EKP object"

    # EKI iterations
    println("EKP inversion error:")
    err = zeros(N_iter)
    for i in 1:N_iter
        ϕ_i = get_ϕ_final(priors, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space    
        g_ens = run_G_ensemble(spunup_state, lsettings, ϕ_i, window, length(y))
        EKP.update_ensemble!(ekiobj, g_ens)
    end
    @info "Calibrated parameters with EKP"
    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(true_params)

    println("\nEKI results:")
    println(get_ϕ_mean_final(priors, ekiobj))

    u_stored = get_u(ekiobj, return_array = false)
    g_stored = get_g(ekiobj, return_array = false)

    @save output_directory * "parameter_storage.jld2" u_stored
    @save output_directory * "output_storage.jld2" g_stored Γ

    # Plots
    fig = Figure(size = (450, 450))
    aprior = Axis(fig[1, 1][1, 1])
    apost = Axis(fig[2, 1][1, 1])
    g_prior = get_g(ekiobj, 1)
    g_post = get_g_final(ekiobj)
    data_std = sqrt.([Γ[i, i] for i in 1:size(Γ, 1)])
    data_dim = length(y)
    dplot = 1:data_dim
    lines!(aprior, dplot, sqrt_inv_Γ * y, color = (:black, 0.5), label = "data") #plots each row as new plot
    lines!(apost, dplot, sqrt_inv_Γ * y, color = (:black, 0.5), label = "data") #plots each row as new plot
    band!(
        aprior,
        dplot,
        sqrt_inv_Γ * (y_mean - 2 * data_std)[:],
        sqrt_inv_Γ * (y_mean + 2 * data_std)[:],
        color = (:grey, 0.2),
    ) #estimated 2*std bands about a mean 
    band!(
        apost,
        dplot,
        sqrt_inv_Γ * (y_mean - 2 * data_std)[:],
        sqrt_inv_Γ * (y_mean + 2 * data_std)[:],
        color = (:grey, 0.2),
    )
    for idx in 1:N_ens
        lines!(aprior, dplot, sqrt_inv_Γ * g_prior[:, idx], color = :orange, alpha = 0.1, label = "prior") #plots each row as new plot
        lines!(apost, dplot, sqrt_inv_Γ * g_post[:, idx], color = :blue, alpha = 0.1, label = "posterior") #plots each row as new plot
    end
    axislegend(aprior, merge = true, unique = true)
    axislegend(apost, merge = true, unique = true)

    # save
    save(joinpath(output_directory, case * "_datasamples-prior-post.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_datasamples-prior-post.pdf"), fig, pt_per_unit = 3)

    @info "plotted results of perfect experiment"

    ###
    ### (b) Repeat exp (a) with UpdateGroups
    ###

    # We see that 
    bs = Int64(size(Γ, 1) / 5) # known block structure

    # recall the parameters are
    # F[1:3],G,h,c,b
    # and the data blocks are
    # <X>, <Y>, <X^2>, <Y^2>, <XY>  X-slow, Y-fast

    # F(3) -> <X>, <X^2>, 
    #    group_slow = UpdateGroup(collect(1:4), reduce(vcat, [collect(1:bs), collect(2 * bs + 1:3 * bs), collect(4*bs+1:5*bs)]))
    group_slow = UpdateGroup(collect(1:3), reduce(vcat, [collect(1:bs), collect((2 * bs + 1):(3 * bs))]))#, collect(4*bs+1:5*bs)]))
    # G,h,c,b -> <Y>, <Y^2>,<XY>
    #group_fast = UpdateGroup(collect(4:7), collect(1:(5 * bs)))
    group_fast = UpdateGroup(collect(4:7), reduce(vcat, [collect((bs + 1):(2 * bs)), collect((3 * bs + 1):(5 * bs))]))




    ekiobj_grouped = EKP.EnsembleKalmanProcess(
        initial_params,
        y,
        Γ,
        Inversion(),
        localization_method = Localizers.NoLocalization(),
        scheduler = DataMisfitController(terminate_at = 1e4),
        failure_handler_method = SampleSuccGauss(),
        verbose = true,
        update_groups = [group_slow, group_fast],
    )
    @info "Built grouped EKP object"

    # EKI iterations
    println("EKP inversion error:")
    err = zeros(N_iter)
    for i in 1:N_iter
        ϕ_i = get_ϕ_final(priors, ekiobj_grouped) # the `ϕ` indicates that the `params_i` are in the constrained space    
        g_ens = run_G_ensemble(spunup_state, lsettings, ϕ_i, window, length(y))
        EKP.update_ensemble!(ekiobj_grouped, g_ens)
    end
    @info "Calibrated parameters with grouped EKP"
    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(true_params)

    println("\nEKI results:")
    println(get_ϕ_mean_final(priors, ekiobj_grouped))

    u_stored = get_u(ekiobj_grouped, return_array = false)
    g_stored = get_g(ekiobj_grouped, return_array = false)

    @save output_directory * "parameter_storage_grouped.jld2" u_stored
    @save output_directory * "output_storage_grouped.jld2" g_stored


    # Plots
    fig = Figure(size = (450, 450))
    aprior = Axis(fig[1, 1][1, 1])
    apost = Axis(fig[2, 1][1, 1])
    g_prior = get_g(ekiobj_grouped, 1)
    g_post = get_g_final(ekiobj_grouped)
    data_std = sqrt.([Γ[i, i] for i in 1:size(Γ, 1)])
    data_dim = length(y)
    dplot = 1:data_dim
    lines!(aprior, dplot, sqrt_inv_Γ * y, color = (:black, 0.5), label = "data") #plots each row as new plot
    lines!(apost, dplot, sqrt_inv_Γ * y, color = (:black, 0.5), label = "data") #plots each row as new plot
    band!(
        aprior,
        dplot,
        sqrt_inv_Γ * (y_mean - 2 * data_std)[:],
        sqrt_inv_Γ * (y_mean + 2 * data_std)[:],
        color = (:grey, 0.2),
    ) #estimated 2*std bands about a mean 
    band!(
        apost,
        dplot,
        sqrt_inv_Γ * (y_mean - 2 * data_std)[:],
        sqrt_inv_Γ * (y_mean + 2 * data_std)[:],
        color = (:grey, 0.2),
    )
    for idx in 1:N_ens
        lines!(aprior, dplot, sqrt_inv_Γ * g_prior[:, idx], color = :orange, alpha = 0.1, label = "prior") #plots each row as new plot
        lines!(apost, dplot, sqrt_inv_Γ * g_post[:, idx], color = :blue, alpha = 0.1, label = "posterior") #plots each row as new plot
    end
    axislegend(aprior, merge = true, unique = true)
    axislegend(apost, merge = true, unique = true)

    # save
    save(joinpath(output_directory, case * "_datasamples-prior-post-grouped.png"), fig, px_per_unit = 3)
    save(joinpath(output_directory, case * "_datasamples-prior-post-grouped.pdf"), fig, pt_per_unit = 3)

    @info "plotted results of perfect experiment"

end

main()
