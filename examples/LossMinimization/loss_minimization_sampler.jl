using Distributions
using LinearAlgebra
using Random
using Plots

struct EnsembleKalmanSampler{FT}
    obs_mean::Vector{FT} #vector of the observed vector size [N_obs]
    obs_noise_cov::Array{FT,2} #covariance matrix of the observational noise, of size [N_obs × N_obs]
    prior_mean::Vector{FT}
    prior_cov::Array{FT, 2}
end

function update_ensemble(
    ekp::EnsembleKalmanSampler,
    u::AbstractArray,
    g::AbstractArray;
)
    # #catch works when g_in non-square 
    # if !(size(g_in)[2] == ekp.N_ens) 
    #      throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    # end

    # u: N_par × N_ens, g: N_obs × N_ens
    N_par, N_ens = size(u)
    N_obs = size(g, 1)
   
    u_mean = mean(u, dims=2) # u_mean: N_par × 1
    g_mean = mean(g, dims=2) # g_mean: N_obs × 1
    g_cov = cov(g, g, dims = 2, corrected=false) # g_cov: N_obs × N_obs
    u_cov = cov(u, u, dims = 2, corrected=false) # u_cov: N_par × N_par

    # Building tmp matrices for EKS update
    E = g .- g_mean # N_obs × N_ens
    R = g .- ekp.obs_mean # N_obs × N_ens
    D = (1/N_ens) * (E' * (ekp.obs_noise_cov \ R)) # D: N_ens × N_ens
    Δt = 1/(norm(D) + 1e-8)
    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, N_par, N_par) + Δt * (ekp.prior_cov \ u_cov)') \
                  (u
                    .- Δt * (u .- u_mean) * D
                    .+ Δt * u_cov * (ekp.prior_cov \ ekp.prior_mean)
                  )
    u_updated = implicit + sqrt(2*Δt) * rand(noise, N_ens)

    # Calculate error
    mean_g = dropdims(mean(g, dims=2), dims=2)
    diff = ekp.obs_mean - mean_g
    err = dot(diff, ekp.obs_noise_cov \ diff)

    return u_updated, err
end

let
    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Set up observational noise 
    n_obs = 1 # Number of synthetic observations from G(u)
    noise_level = 1e-6 # Defining the observation noise level
    Γy = noise_level * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations       
    noise = MvNormal(zeros(n_obs), Γy)

    # Set up the loss function (unique minimum)
    function G(u)
        return [sqrt((u[1]-1)^2 + (u[2]+1)^2)]
    end
    u_star = [1.0, -1.0] # Loss Function Minimum
    y_obs  = G(u_star) + 0 * rand(noise) 

    # Set up prior
    prior_mean = [10.0, 10.0]
    prior_cov = 1 * Matrix(I, length(prior_mean), length(prior_mean))
    prior = MvNormal(prior_mean, prior_cov)

    # Set up optimizer
    ekiobj = EnsembleKalmanSampler{Float64}(y_obs, Γy, prior_mean, prior_cov)

    # Do optimization loop
    N_ens = 50  # number of ensemble members
    N_iter = 100 # number of EKI iterations
    ensemble = rand(prior, N_ens)
    storage_g = []
    storage_u = [copy(ensemble)]
    storage_e = []
    for i in 1:N_iter
        evaluations = hcat(map(G, eachcol(ensemble))...)
        ensemble, err = update_ensemble(ekiobj, ensemble, evaluations)
        push!(storage_u, copy(ensemble))
        push!(storage_g, copy(evaluations))
        push!(storage_e, err)
    end

    # Do plotting
    u_init = storage_u[1]
    u1_min = minimum(minimum(u[1,:]) for u in storage_u)
    u1_max = maximum(maximum(u[1,:]) for u in storage_u)
    u2_min = minimum(minimum(u[2,:]) for u in storage_u)
    u2_max = maximum(maximum(u[2,:]) for u in storage_u)
    xlims = (u1_min, u1_max)
    ylims = (u2_min, u2_max)
    for (i, u) in enumerate(storage_u)
        p = plot(u[1,:], u[2,:], seriestype=:scatter, xlims = xlims, ylims = ylims)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red, label = false,
            title = "EKI iteration = " * string(i)
            )
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")
        display(p)
        sleep(0.1)
    end
end