using Distributions
using LinearAlgebra
using Random
using Test

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
@testset "EnsembleKalmanProcessModule" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    ### Generate data from a linear model: a regression problem with n_par parameters
    ### and n_obs. G(u) = A \times u, where A : R^n_par -> R
    n_obs = 10                  # Number of synthetic observations from G(u)
    n_par =  2                  # Number of parameteres
    u_star = [-1., 2.]          # True parameters
    noise_level = .1            # Defining the observation noise level
    Γy = noise_level * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations
    noise = MvNormal(zeros(n_obs), Γy)

    C = [1 -.9; -.9 1]          # Correlation structure for linear operator
    A = rand(MvNormal(zeros(2,), C), n_obs)'    # Linear operator in R^{n_obs \times n_par}

    @test size(A) == (n_obs, n_par)

    y_star = A * u_star
    y_obs  = y_star + rand(noise)

    @test size(y_star) == (n_obs,)

    #### Define linear model
    function G(u)
        A * u
    end

    @test norm(y_star - G(u_star)) < n_obs * noise_level^2

    #### Define prior information on parameters
    prior_distns = [Parameterized(Normal(1., sqrt(2))),
              Parameterized(Normal(1., sqrt(2)))]
    constraints = [[no_constraint()], [no_constraint()]]
    prior_names = ["u1","u2"]
    prior = ParameterDistribution(prior_distns, constraints, prior_names)
    
    prior_mean = reshape(get_mean(prior),:)

    # Assuming independence of u1 and u2
    prior_cov = get_cov(prior)#convert(Array, Diagonal([sqrt(2.), sqrt(2.)]))

    ###
    ###  Calibrate (1): Ensemble Kalman Inversion
    ###

    N_ens = 50 # number of ensemble members
    N_iter = 20 # number of EKI iterations
    initial_ensemble = EnsembleKalmanProcessModule.construct_initial_ensemble(prior,N_ens;
                                                    rng_seed=rng_seed)
    @test size(initial_ensemble) == (n_par, N_ens)

    ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                       y_obs, Γy, Inversion())

    # Find EKI timestep
    g_ens = G(get_u_final(ekiobj))
    @test size(g_ens) == (n_obs, N_ens)
    # as the columns of g are the data, this should throw an error
    g_ens_t = permutedims(g_ens, (2,1))
    @test_throws DimensionMismatch find_ekp_stepsize(ekiobj, g_ens_t)
    Δ = find_ekp_stepsize(ekiobj, g_ens)
    @test Δ ≈ 0.0625
    
    # EKI iterations
    params_i_vec = []
    g_ens_vec = []
    for i in 1:N_iter
        params_i = get_u_final(ekiobj)
        push!(params_i_vec,params_i)
        g_ens = G(params_i)
        push!(g_ens_vec, g_ens)
        if i == 1
            g_ens_t = permutedims(g_ens, (2,1))
            @test_throws DimensionMismatch EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens_t)
        end
        EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)
    end
    push!(params_i_vec,get_u_final(ekiobj))
    
    @test get_u_prior(ekiobj) == params_i_vec[1]
    @test get_u(ekiobj) == params_i_vec
    @test get_g(ekiobj) == g_ens_vec
    @test get_g_final(ekiobj) == g_ens_vec[end]
    @test get_error(ekiobj) == ekiobj.err
    
    # EKI results: Test if ensemble has collapsed toward the true parameter 
    # values
    eki_final_result = vec(mean(get_u_final(ekiobj), dims=2))
    # @test norm(u_star - eki_final_result) < 0.5

    # Plot evolution of the EKI particles
    eki_final_result = vec(mean(get_u_final(ekiobj), dims=2))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(get_u_prior(ekiobj)[1,:], get_u_prior(ekiobj)[2,:], seriestype=:scatter)
        plot!(get_u_final(ekiobj)[1, :],  get_u_final(ekiobj)[2,:], seriestype=:scatter)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red)
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
        savefig(p, "EKI_test.png")
    end

    ###
    ###  Calibrate (2): Ensemble Kalman Sampleer
    ###
    eksobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble,
                      y_obs, Γy,
                      Sampler(prior_mean, prior_cov))

    # EKS iterations
    for i in 1:N_iter
        params_i = get_u_final(eksobj)
        g_ens = G(params_i)
        if i == 1
            g_ens_t = permutedims(g_ens, (2,1))
            @test_throws DimensionMismatch EnsembleKalmanProcessModule.update_ensemble!(eksobj, g_ens_t)
        end

        EnsembleKalmanProcessModule.update_ensemble!(eksobj, g_ens)
    end

    # Plot evolution of the EKS particles
    eks_final_result = vec(mean(get_u_final(eksobj), dims=2))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(get_u_prior(eksobj)[1,:], get_u_prior(eksobj)[2,:], seriestype=:scatter)
        plot!(get_u_final(eksobj)[1, :],  get_u_final(eksobj)[2,:], seriestype=:scatter)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red)
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
        savefig(p, "EKS_test.png")
    end

    posterior_cov_inv = (A' * (Γy\A) + 1 * Matrix(I, n_par, n_par)/prior_cov)
    ols_mean          = (A'*(Γy\A)) \ (A'*(Γy\y_obs))
    posterior_mean    = posterior_cov_inv \ ( (A'*(Γy\A)) * ols_mean  + (prior_cov\prior_mean))

    #### This tests correspond to:
    # EKI provides a solution closer to the ordinary Least Squares estimate
    @test norm(ols_mean - eki_final_result) < norm(ols_mean - eks_final_result)
    # EKS provides a solution closer to the posterior mean
    @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - eki_final_result)

    ##### I expect this test to make sense:
    # In words: the ensemble covariance is still a bit ill-dispersed since the
    # algorithm employed still does not include the correction term for finite-sized
    # ensembles.
    @test abs(sum(diag(posterior_cov_inv\cov(get_u_final(eksobj),dims=2))) - n_par) > 1e-5



    ###
    ###  Calibrate (3): Unscented Kalman Inversion
    ###

    N_iter = 20 # number of UKI iterations
    α_reg =  1.0
    update_freq = 0
    process = Unscented(prior_mean, prior_cov, length(y_star),  α_reg, update_freq)
    ukiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(y_star, Γy, process)

    # UKI iterations
    params_i_vec = []
    g_ens_vec = []
    for i in 1:N_iter
        params_i = get_u_final(ukiobj)
        push!(params_i_vec,params_i)
        g_ens = G(params_i)
        push!(g_ens_vec, g_ens)
        if i == 1
            g_ens_t = permutedims(g_ens, (2,1))
            @test_throws DimensionMismatch EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens_t)
        end
        EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens)
    end
    push!(params_i_vec,get_u_final(ukiobj))
    
    @test get_u_prior(ukiobj) == params_i_vec[1]
    @test get_u(ukiobj) == params_i_vec
    @test get_g(ukiobj) == g_ens_vec
    @test get_g_final(ukiobj) == g_ens_vec[end]
    @test get_error(ukiobj) == ukiobj.err
    
    # UKI results: Test if ensemble has collapsed toward the true parameter 
    # values
    uki_final_result = get_u_mean_final(ukiobj)
    @test norm(u_star - uki_final_result) < 0.5

    if TEST_PLOT_OUTPUT
        gr()
        θ_mean_arr = hcat(ukiobj.process.u_mean...)
        N_θ = length(ukiobj.process.u_mean[1])
        θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
        for i = 1:N_iter+1
            for j = 1:N_θ
                θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j,j])
            end
        end

        ites = Array(LinRange(1, N_iter+1, N_iter+1))
        p = plot(ites,grid=false, θ_mean_arr[1,:], yerror=3.0*θθ_std_arr[1,:],  label="u1")
        plot!(ites, fill(u_star[1], N_iter+1), linestyle=:dash, linecolor=:grey,label=nothing)
        plot!(ites,grid=false, θ_mean_arr[2,:], yerror=3.0*θθ_std_arr[2,:], label="u2", xaxis="Iterations")
        plot!(ites, fill(u_star[2], N_iter+1), linestyle=:dash, linecolor=:grey,label=nothing)
        savefig(p, "UKI_test.png")
    end

end
