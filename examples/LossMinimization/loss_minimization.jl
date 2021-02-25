using Distributions
using LinearAlgebra
using Random
using Plots
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributionStorage

# Seed for pseudo-random number generator for reproducibility
rng_seed = 41
Random.seed!(rng_seed)

# Number of synthetic observations from G(u)
n_obs = 1
# Defining the observation noise level
noise_level =  1e-8   
# Independent noise for synthetic observations       
Γy = noise_level * Matrix(I, n_obs, n_obs) 
noise = MvNormal(zeros(n_obs), Γy)

# Loss Function (unique minimum)
function G(u)
    return [sqrt((u[1]-1)^2 + (u[2]+1)^2)]
end

# Loss Function Minimum
u_star = [1.0, -1.0]
y_obs  = G(u_star) + 0 * rand(noise) 

# Define Prior
prior_distns = [Parameterized(Normal(0., sqrt(1))),
                Parameterized(Normal(-0., sqrt(1)))]
constraints = [[no_constraint()], [no_constraint()]]
prior_names = ["u1", "u2"]
prior = ParameterDistribution(prior_distns, constraints, prior_names)
prior_mean = reshape(get_mean(prior),:)
prior_cov = get_cov(prior)

# Calibrate
N_ens = 50  # number of ensemble members
N_iter = 20 # number of EKI iterations
initial_ensemble = EnsembleKalmanProcesses.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())
#
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    g_ens = hcat([G(params_i[:,i]) for i in 1:N_ens]...)
    EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens)
end

u_init = get_u_prior(ekiobj)

for i in 1:N_iter
    u_i = get_u(ekiobj,i)
    p = plot(u_i[1,:], u_i[2,:], seriestype=:scatter, xlims = extrema(u_init[1,:]), ylims = extrema(u_init[2,:]))
    plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
        linestyle=:dash, linecolor=:red, label = false,
        title = "EKI iteration = " * string(i)
        )
    plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")
    display(p)
    sleep(0.1)
end

##
rng_seed = 10 # 10 converges to one minima 100 converges to the other

# Loss Function (two minima)
function G(u)
    return [abs((u[1]-1)*(u[1]+1))^2 + (u[2]+1)^2]
end

# Loss Function Minimum
u_star1 = [1.0, -1.0]
u_star2 = [-1.0, -1.0]
G(u_star1)[1] == G(u_star2)[1]
y_obs  = [0.0]

# Define Prior
prior_distns = [Parameterized(Normal(0., sqrt(2))),
                Parameterized(Normal(-0., sqrt(2)))]
constraints = [[no_constraint()], [no_constraint()]]
prior_names = ["u1", "u2"]
prior = ParameterDistribution(prior_distns, constraints, prior_names)
prior_mean = reshape(get_mean(prior),:)
prior_cov = get_cov(prior)

# Calibrate
N_ens = 50  # number of ensemble members
N_iter = 40 # number of EKI iterations
initial_ensemble = EnsembleKalmanProcesses.construct_initial_ensemble(prior, N_ens;
                                                rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble,
                    y_obs, Γy, Inversion())
#
for i in 1:N_iter
    params_i = get_u_final(ekiobj)
    g_ens = hcat([G(params_i[:,i]) for i in 1:N_ens]...)
    EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens)
end

u_init = get_u_prior(ekiobj)
for i in 1:N_iter
    u_i = get_u(ekiobj,i)
    p = plot(u_i[1,:], u_i[2,:], seriestype=:scatter, xlims = (-2,2), ylims = (-2,2))
    plot!([1], xaxis="u1", yaxis="u2", seriestype="vline",
        linestyle=:dash, linecolor=:red, label = false,
        title = "EKI iteration = " * string(i)
        )
    plot!([-1], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optima 1")

    plot!([-1], xaxis="u1", yaxis="u2", seriestype="vline",
        linestyle=:dash, linecolor=:green, label = false,
        title = "EKI iteration = " * string(i)
        )
    plot!([-1], seriestype="hline", linestyle=:dash, linecolor=:green, label = "optima 2")
    display(p)
    sleep(0.1)
end
