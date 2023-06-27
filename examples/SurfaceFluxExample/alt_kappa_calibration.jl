# # Alternative Kappa Calibration Example
# ## Overview
#=
In this example, just like in kappa_calibration.jl, we use the inverse problem to calibrate the von-karman constant, κ in
the equation: u(z) = u^* / κ log (z / z0), 
which represents the wind profile in Monin-Obukhov
Similarity Theory (MOST) formulations. We use the same dataset: https://turbulence.pha.jhu.edu/Channel_Flow.aspx

Instead of using u^* as an observable, we use the dataset's u, and each ensemble member will estimate u
through the profile equation u(z) = u^* / κ log (z / z0).
=#

# ## Prerequisites
#=
[EnsembleKalmanProcess.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl),
=#

# ## Example

# First, we import relevant modules.
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using DelimitedFiles

FT = Float64

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
web_datafile_path = "https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_5200_mean_prof.dat"
localfile = "data/profiles.dat"
Downloads.download(web_datafile_path, localfile)
data_mean_velocity = readdlm("data/profiles.dat", skipstart = 112) ## We skip 72 lines (header) and 40(laminar layer)

web_datafile_path = "https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_5200_mean_stdev.dat"
localfile = "data/vel_stdev.dat"
Downloads.download(web_datafile_path, localfile)
# We skip 72 lines (header) and 40(laminar layer)
data_stdev_velocity = readdlm("data/vel_stdev.dat", skipstart = 112)

# We extract the required info for this problem
u_star_obs = 4.14872e-02 # add noise later
z0 = FT(0.0001)
κ = 0.4

# turn u into distributions
u = data_mean_velocity[:, 3] * u_star_obs
z = data_mean_velocity[:, 1]
u = u[1:(length(u) - 1)] # filter out last element because σᵤ is only of length 727, not 728
z = z[1:(length(z) - 1)]

σᵤ = data_stdev_velocity[:, 4] * u_star_obs
dist_u = Array{Normal{Float64}}(undef, length(u))
for i in 1:length(u)
    dist_u[i] = Normal(u[i], σᵤ[i])
end

# u(z) = u^* / κ log (z / z0)
function physical_model(parameters, inputs)
    κ = parameters[1] # this is being updated by the EKP iterator
    (; u_star_obs, z, z0) = inputs
    u_profile = u_star_obs ./ κ .* log.(z ./ z0)
    return u_profile
end

function G(parameters, inputs, u_profile = nothing)
    if (isnothing(u_profile))
        u_profile = physical_model(parameters, inputs)
    end
    return [maximum(u_profile) - minimum(u_profile), mean(u_profile)]
end

Γ = 0.0001 * I
η_dist = MvNormal(zeros(2), Γ)
noisy_u_profile = [rand(dist_u[i]) for i in 1:length(u)]
y = G(nothing, nothing, noisy_u_profile)

parameters = (; κ)
inputs = (; u_star_obs, z, z0)
# y = G(parameters, inputs) .+ rand(η_dist)

# Assume that users have prior knowledge of approximate truth.
# (e.g. via physical models / subset of obs / physical laws.)
prior_u1 = constrained_gaussian("κ", 0.35, 0.25, 0, Inf);
prior = combine_distributions([prior_u1])

# Set up the initial ensembles
N_ensemble = 5;
N_iterations = 10;

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble);

# Define EKP and run iterative solver for defined number of iterations
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

# Mean values in final ensemble for the two parameters of interest reflect the "truth" within some degree of 
# uncertainty that we can quantify from the elements of `final_ensemble`.
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)
mean(final_ensemble[1, :]) # [param, ens_no]


ENV["GKSwstype"] = "nul"
zrange = z
plot(zrange, noisy_u_profile, c = :black, label = "Truth", linewidth = 2, legend = :bottomright)
plot!(zrange, physical_model(parameters, inputs), c = :green, label = "Model truth", linewidth = 2)# reshape to convert from vector to matrix)
plot!(
    zrange,
    [physical_model(get_ϕ(prior, ensemble_kalman_process, 1)[:, i]..., inputs) for i in 1:N_ensemble],
    c = :red,
    label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble), # reshape to convert from vector to matrix
)
plot!(
    zrange,
    [physical_model(final_ensemble[:, i]..., inputs) for i in 1:N_ensemble],
    c = :blue,
    label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble),
)
xlabel!("Z")
ylabel!("U")
png("profile_plot")
