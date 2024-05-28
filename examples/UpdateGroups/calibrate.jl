
# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using CairoMakie, ColorSchemes
using Random
using JLD2

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

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
    year = 360.0*time_scale_factor
    month = year/12.0*time_scale_factor

    # Fast 
    h_true = 10.0
    c_true = 10.0# timescale difference
    b_true = 10.0
    
    # Slow
    F_true= [8.0, 2.0, 1/year] # (mean,amplitude,frequency) const for now (maybe add seasonal cycle)
    G_true = h_true*c_true/b_true # for consistency in coupling terms
    
    true_params = LParams(F_true, G_true, h_true, c_true, b_true)
    param_names = ["F", "G", "h", "c", "b"]
    
    ###
    ###  Define the parameter priors (needed later for inversion)
    ###
    
    prior_F = constrained_gaussian(param_names[1], 5.0, 5.0, 0, Inf, repeat=3)
    prior_G = constrained_gaussian(param_names[2], 5.0, 5.0, 0, Inf)
    prior_h = constrained_gaussian(param_names[3], 5.0, 5.0, 0, Inf)
    prior_c = constrained_gaussian(param_names[4], 5.0, 5.0, 0, Inf)
    prior_b = constrained_gaussian(param_names[5], 5.0, 5.0, 0, Inf)
priors = combine_distributions([prior_F, prior_G, prior_h, prior_c, prior_b])
    
    ###
    ### Configuration of the forward map
    ###
    
    ## State evolution
    N = 40 # state size
    dt = 0.005 # integration dt
    
    
    ## Spin up
    spinup = year
    initial = rand(Normal(0.0,0.01),2*N) # IC for fast and slow
    
    lsettings_spinup = LSettings(N, dt, spinup, initial)

    ## Gather statistics
    # e.g. run for 2 years, using last year for data, (& batched monthly)
    t_solve = 2*year
    window_start = t_solve - year
    window_end = t_solve
    subwindow_size = Int64(floor(month/ dt))
    slide_or_batch = "batch" #sliding window or disjoint batches
    
    lsettings = LSettings(N, dt, t_solve, nothing)
    window = ProcessingWindow(window_start, window_end, subwindow_size, slide_or_batch)
    
###
### Spin up and plot
###

spunup_state, t = lorenz_solve(lsettings_spinup, true_params)
@info "spin-up complete"

fig = Figure(resolution = (900, 450))

afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

    ss_freq = length(t)/2010.0
    ss_rate = Int64(ceil(ss_freq))
    tplot = t[10*ss_rate+1:ss_rate:end]
hm = heatmap!(afast, tplot, 1:N, spunup_state[N+1:2*N,10*ss_rate+1:ss_rate:end]', colormap =  :Oranges)
Colorbar(fig[1, 1][1, 2], hm)
hm = heatmap!(aslow, tplot, 1:N, spunup_state[1:N,10*ss_rate+1:ss_rate:end]', colormap = :Blues)
Colorbar(fig[2, 1][1, 2], hm)

# save
save(joinpath(output_directory, case * "_spinup_allstate.png"), fig, px_per_unit = 3)
save(joinpath(output_directory, case * "_spinup_allstate.pdf"), fig, pt_per_unit = 3)


fig = Figure(resolution = (900, 450))

afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

lines!(afast, tplot, spunup_state[N+1,10*ss_rate+1:ss_rate:end], color = :red)
lines!(aslow, tplot, spunup_state[1,10*ss_rate+1:ss_rate:end], color = :blue)

    # save
save(joinpath(output_directory, case * "_spinup_state1.png"), fig, px_per_unit = 3)
save(joinpath(output_directory, case * "_spinup_state1.pdf"), fig, pt_per_unit = 3)
@info "plotted spin-up"    

###
### Generate perfect-model data
###

    #estimate covariance
    n_sample_cov = 40
    
    new_state, new_t = lorenz_solve(spunup_state, lsettings, true_params)
    data_sample = process_trajectory_to_data(window, new_state, new_t)
    
    data_samples = zeros(size(data_sample,1),n_sample_cov)
    data_samples[:,1] = data_sample
    
    fig = Figure(resolution = (900, 450))
    afast = Axis(fig[1, 1][1, 1], xlabel = "time", ylabel = "fast")
    aslow = Axis(fig[2, 1][1, 1], xlabel = "time", ylabel = "slow")

    for i = 2:n_sample_cov
        # extend trajectory from end of new_state
        new_state, new_t = lorenz_solve(new_state, lsettings, true_params) #integrate another window from last state
        # calculate data sample
        data_samples[:,i] = process_trajectory_to_data(window, new_state, new_t) # process the window 

        #plot trajectory
        ss_freq = length(new_t)/2010.0
        ss_rate = Int64(ceil(ss_freq))
        tplot = new_t[Int64(floor(length(new_t)/2)):ss_rate:end]

        lines!(afast, tplot, new_state[N+1,Int64(floor(length(new_t)/2)):ss_rate:end], color = :red, alpha=0.1)
        lines!(aslow, tplot, new_state[1,Int64(floor(length(new_t)/2)):ss_rate:end], color = :blue, alpha=0.1,  label="$(n_sample_cov) samples")
        axislegend(aslow, merge=true, unique=true)
    end
    # save
    save(joinpath(output_directory, case * "_datatrajectory_state1.png"), fig, px_per_unit = 3)
save(joinpath(output_directory, case * "_datatrajectory_state1.pdf"), fig, pt_per_unit = 3)
    
    Γ = cov(data_samples,dims=2) # estimate covariance from samples
    y = data_samples[:,shuffle(rng,1:n_sample_cov)[1]] # random data point as the data

fig = Figure(resolution = (450, 450))
aΓ = Axis(fig[1, 1][1, 1])
adata = Axis(fig[2, 1][1, 1])

heatmap!(aΓ,Γ)
series!(adata, data_samples', solid_color= :black) #plots each row as new plot

    # save
    save(joinpath(output_directory, case * "_datasamples.png"), fig, px_per_unit = 3)
save(joinpath(output_directory, case * "_datasamples.pdf"), fig, pt_per_unit = 3)

    @info "constructed and plotted perfect experiment data and noise"    
    

# may need to condition Gamma for posdef etc. or add shrinkage

###
### Configure ensemble inversion
###






                          
end

main()
