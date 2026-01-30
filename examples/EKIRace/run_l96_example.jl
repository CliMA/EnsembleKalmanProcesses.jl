# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux
using BSON

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include("Lorenz96.jl") # Contains Lorenz 96 source code

########################################################################
############### Choose problem type and structure ######################
########################################################################

cases = ["const-force", "vec-force", "flux-force"] # problem types
# User specifications
case = cases[2] # choose problem type
N_ens_sizes = [80, 90, 100] # list of number of ensemble members (should be problem dependent)
N_iter = 20 # maximum number of EKI iterations allowed
target_rmse = 1.0 # target RMSE
n_repeats = 4
rng_seeds = randperm(1_000_000)[1:n_repeats] # list of random seeds
@info "Running $case case"
@info "Maximum number of EKI iterations: $N_iter"
@info "RMSE target: $target_rmse"
# saved and loaded for plotting etc.
configuration = Dict(
    "case" => case,
    "N_iter" => N_iter,
    "N_ens_sizes" => N_ens_sizes,
    "target_rmse" => target_rmse,
    "rng_seeds" => rng_seeds,
)


if case == "const-force"
    nx = 40  #dimensions of parameter vector
    nu = 1
    phi = ConstantEMC(8.0) # forcing
    phi_structure = nothing
    sample_range = nothing
    prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
    T = 14.0
    inff = 2
elseif case == "vec-force"
    nx = 40  # dimensions of parameter vector
    nu = nx
    sinusoid = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)
    phi = VectorEMC(sinusoid)
    phi_structure = nothing
    sample_range = nothing
    pl, psig = 2.0, 3.0
    prior_cov = zeros(nx, nx) # prior covariance
    for ii in 1:nx
        for jj in 1:nx
            prior_cov[ii, jj] = psig^2 * exp(-abs(ii - jj) / pl)
        end
    end
    prior_mean = 8.0 * ones(nx) # prior mean
    #Creating prior distribution
    distribution = Parameterized(MvNormal(prior_mean, prior_cov))
    constraint = repeat([no_constraint()], 40)
    name = "ml96_prior"
    prior = ParameterDistribution(distribution, constraint, name)
    T = 54.0
    inff = 2
elseif case == "flux-force"
    nx = 100  #dimensions of parameter vector
    nu = 61
    true_sinusoid(x) = 8 .+ 6 * sin.((4 * pi * x) / 10)
    x_train = collect(-5.0:0.01:5.0)
    y_train = true_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
    phi_structure = Chain(Dense(1 => 20, tanh), Dense(20 => 1))
    true_model, _ = train_network(phi_structure, x_train, y_train)
    sample_range = Float32.(collect(-5.0:0.1:4.9))
    phi = FluxEMC(true_model, sample_range)

    prior_sinusoid(x) = 8.02 .+ 6.5 * sin.(1.02 * (4 * pi * x) / 10 + 0.2)
    prior_train = prior_sinusoid.(x_train) .+ 0.2 .* randn(length(x_train))
    prior_model, prior_mean = train_network(phi_structure, x_train, prior_train)

    y_pred = true_model(reshape(sample_range, 1, :))
    prior_pred = prior_model(reshape(sample_range, 1, :))
    # Plot for visualizing training results
    # p = plot(sample_range, true_sinusoid.(sample_range), label="True function", lw=2, color=:blue)
    # scatter!(p, x_train, y_train, label="Training data", ms=1, color=:red, alpha=0.5)
    # plot!(p, sample_range, y_pred[1,:], label="Predicted function", lw=2, color=:green)
    # xlabel!("x")
    # ylabel!("y")
    # title!("Offline 1D DNN")
    # display(p)

    # p = plot(sample_range, prior_sinusoid.(sample_range), label="True prior function", lw=2, color=:blue)
    # scatter!(p, x_train, prior_train, label="Training data", ms=1, color=:red, alpha=0.5)
    # plot!(p, sample_range, prior_pred[1,:], label="Predicted prior function", lw=2, color=:green)
    # xlabel!("x")
    # ylabel!("y")
    # title!("Offline 1D DNN")
    # display(p)

    # p = plot(sample_range, prior_sinusoid.(sample_range), label="Prior function", lw=2, color=:blue)
    # plot!(p, sample_range, true_sinusoid.(sample_range), label="True function", lw=2, color=:green)
    # xlabel!("x")
    # ylabel!("y")
    # title!("Offline 1D DNN")
    # display(p)
    prior_cov = (0.1^2) * I(length(prior_mean))
    distribution = Parameterized(MvNormal(prior_mean, prior_cov))
    constraint = repeat([no_constraint()], 61)
    name = "l96_nn_prior"
    prior = ParameterDistribution(distribution, constraint, name)
    T = 54.0
    inff = 2.5

end


########################################################################
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

output_dir = joinpath(@__DIR__, "output")
if !isdir(output_dir)
    mkdir(output_dir)
end

ny = nx * 2   #number of data points

#Creating my sythetic data
prelim_file = joinpath(output_dir, "l96_computed_preliminaries_$(case).jld2")
if isfile(prelim_file)
    loaded_data = JLD2.load(prelim_file)
    x0 = loaded_data["x0"]
    y = loaded_data["y"]
    ic_cov_sqrt = loaded_data["ic_cov_sqrt"]
    R = loaded_data["R"]
    R_inv_var = loaded_data["R_inv_var"]
    lorenz_config_settings = loaded_data["lorenz_config_settings"]
    observation_config = loaded_data["observation_config"]

    @info "loaded precomputed preliminary quantities from $(prelim_file)"
else
    t = 0.01  #time step
    T_long = 1000.0  #total time 
    lorenz_config_settings = LorenzConfig(t, T)
    @info "Initial spin up stage: Running $( (T_long)) time units"
    picking_initial_condition = LorenzConfig(t, T_long)
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx) # initial condition for spinning up Lorenz system
    x_spun_up = lorenz_solve(phi, x_initial, picking_initial_condition) # spinning up Lorenz system

    x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data

    # construct how we compute Observations
    T_start = 4.0  #2*max
    T_end = T
    observation_config = ObservationConfig(T_start, T_end)
    @info "Compute synthetic data: Running $(T) time units"
    y = lorenz_forward(phi, x0, lorenz_config_settings, observation_config) # synthetic data

    #Observation covariance R
    window = T_end - T_start
    T_R = 10 * window * ny + T_start
    R_config = LorenzConfig(t, T_R)
    @info "Estimating noise covariance: Running $(T_R) time units"
    R_run = lorenz_solve(phi, x_initial, R_config)

    R_sample_size = Int(ceil(10 * ny))
    R_samples = zeros(ny, R_sample_size)
    for ii in 1:R_sample_size
        local_obs_config = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
        R_samples[:, ii] = stats(R_run, R_config, local_obs_config)
    end
    R = cov(R_samples, dims = 2) * inff
    R_sqrt = sqrt(R)
    R_inv_var = sqrt(inv(R))

    # Need a way to perturb the initial condition when doing the EKI updates
    # Solving for initial condition perturbation covariance
    covT = 2000.0  #time to simulate to calculate a covariance matrix of the system
    @info "Estimating initial condition covariance: Running $(covT) time units"
    cov_solve = lorenz_solve(phi, x0, LorenzConfig(t, covT))
    ic_cov = 0.1 * cov(cov_solve, dims = 2)
    ic_cov_sqrt = sqrt(ic_cov)

    JLD2.save(
        prelim_file,
        "x0",
        x0,
        "y",
        y,
        "lorenz_config_settings",
        lorenz_config_settings,
        "observation_config",
        observation_config,
        "ic_cov_sqrt",
        ic_cov_sqrt,
        "R",
        R,
        "R_inv_var",
        R_inv_var,
    )
    @info "saving computed quantities in $prelim_file"
end


########################################################################
########################### Running EKI Race ###########################
########################################################################

# Counters
conv_alg_iters = zeros(4, length(N_ens_sizes), length(rng_seeds)) #count how many iterations it takes to converge (per algorithm, per rand seed, per ense size)
final_parameters = zeros(4, length(N_ens_sizes), length(rng_seeds), nu)
final_model_output = zeros(4, length(N_ens_sizes), length(rng_seeds), ny)

method_names = [
    ("Inversion(prior)", "TEKI"),
    ("TransformInversion(prior)", "ETKI"),
    ("GaussNewtonInversion(prior)", "GNKI"),
    ("Unscented(prior; impose_prior=true)", "UKI"),
]

for (rr, rng_seed) in enumerate(rng_seeds)
    @info "Random seed: $(rng_seed)"
    rng = MersenneTwister(rng_seed)

    for (ee, N_ens) in enumerate(N_ens_sizes)
        # initial parameters: N_params x N_ens
        initial_params = construct_initial_ensemble(rng, prior, N_ens)
        methods = [
            Inversion(prior),
            TransformInversion(prior),
            GaussNewtonInversion(prior),
            Unscented(prior; impose_prior = true),
        ]

        @info "Ensemble size: $(N_ens)"
        for (kk, method) in enumerate(methods)
            @info "Method: $(nameof(typeof(method)))"
            if isa(method, Unscented)
                ekpobj = EKP.EnsembleKalmanProcess(
                    y,
                    R,
                    deepcopy(method);
                    rng = copy(rng),
                    verbose = true,
                    accelerator = DefaultAccelerator(),
                    localization_method = NoLocalization(),
                    scheduler = DefaultScheduler(),
                )
            else
                ekpobj = EKP.EnsembleKalmanProcess(
                    initial_params,
                    y,
                    R,
                    deepcopy(method);
                    rng = copy(rng),
                    verbose = true,
                    accelerator = DefaultAccelerator(),
                    localization_method = NoLocalization(),
                    scheduler = DefaultScheduler(),
                )
            end
            Ne = get_N_ens(ekpobj)

            count = 0
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, ekpobj)

                # Calculating RMSE_e
                ens_mean = mean(params_i, dims = 2)[:]
                # forcing = build_forcing(ens_mean, phi_structure)
                forcing = build_forcing(phi, ens_mean, phi_structure, sample_range)
                G_ens_mean = lorenz_forward(
                    forcing,
                    x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, 1),
                    lorenz_config_settings,
                    observation_config,
                )
                RMSE_e = norm(R_inv_var * (y - G_ens_mean[:])) / sqrt(size(y, 1))
                @info "RMSE (at G(u_mean)): $(RMSE_e)"
                # Convergence criteria
                if RMSE_e < target_rmse
                    conv_alg_iters[kk, ee, rr] = count * Ne
                    final_parameters[kk, ee, rr, :] = ens_mean
                    final_model_output[kk, ee, rr, :] = G_ens_mean
                    break
                end

                # If RMSE convergence criteria is not satisfied 
                G_ens = hcat(
                    [
                        lorenz_forward(
                            build_forcing(phi, params_i[:, j], phi_structure, sample_range),
                            (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j],
                            lorenz_config_settings,
                            observation_config,
                        ) for j in 1:Ne
                    ]...,
                )
                # Update 
                EKP.update_ensemble!(ekpobj, G_ens)
                count = count + 1

                # # Calculate RMSE_f to the mean G(u) if desired
                # RMSE_f = sqrt(get_error_metrics(ekpobj)["loss"][end])
                # @info "RMSE (at mean(G(u)): $(RMSE_f)"
                # # Convergence criteria
                # if RMSE_f < target_rmse
                #     conv_alg_iters[kk, ee, rr] = count * Ne
                #     final_parameters[kk, ee, rr, :] = ens_mean
                #     final_model_output[kk, ee, rr, :] = G_ens_mean
                #     break
                # end
            end

            final_ensemble = get_ϕ_final(prior, ekpobj)
        end
    end
end


# Saving data:
using Dates
date_of_exp = today()
data_filename = joinpath(output_dir, "l96_output_$(case)_$(today()).jld2")
JLD2.save(
    data_filename,
    "configuration",
    configuration,
    "method_names",
    method_names,
    "conv_alg_iters",
    conv_alg_iters,
    "final_parameters",
    final_parameters,
    "final_model_output",
    final_model_output,
)
