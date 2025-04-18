if TEST_PLOT_OUTPUT
    # Fix seed, so the plots do not differ from run to run
    import Random
    rng_seed = 1234
    rng = Random.MersenneTwister(rng_seed)

    using Test
    using CairoMakie

    using LinearAlgebra
    using EnsembleKalmanProcesses
    using EnsembleKalmanProcesses.ParameterDistributions

    @testset "Helper functions" begin
        prior_u1 = constrained_gaussian("positive_with_mean_1", 2, 1, 0, Inf)
        prior_u2 = constrained_gaussian("four_with_spread_3", 0, 5, -Inf, Inf, repeats = 3)
        prior_u3 = constrained_gaussian("positive_with_mean_1", 0, 3, -Inf, Inf)
        prior = combine_distributions([prior_u1, prior_u2, prior_u3])

        # Since these functions are not exported, we need to access the
        # functions like this
        ext = Base.get_extension(EnsembleKalmanProcesses, :EnsembleKalmanProcessesMakieExt)
        @test ext._get_dim_of_dist(prior, 1) == 1
        @test ext._get_dim_of_dist(prior, 2) == 1
        @test ext._get_dim_of_dist(prior, 3) == 2
        @test ext._get_dim_of_dist(prior, 4) == 3
        @test ext._get_dim_of_dist(prior, 5) == 1
        @test_throws ErrorException ext._get_dim_of_dist(prior, 6)

        @test ext._get_prior_name(prior, 1) == "positive_with_mean_1"
        @test ext._get_prior_name(prior, 2) == "four_with_spread_3"
        @test ext._get_prior_name(prior, 3) == "four_with_spread_3"
        @test ext._get_prior_name(prior, 4) == "four_with_spread_3"
        @test ext._get_prior_name(prior, 5) == "positive_with_mean_1"
        @test_throws ErrorException ext._get_prior_name(prior, 6)

    end
    @testset "Makie ploting" begin
        # Access plots at tmp_dir
        tmp_dir = mktempdir(cleanup = false)
        @info "Tempdir", tmp_dir

        G(u) = [1.0 / abs(u[1]), sum(u[2:3]), u[3], u[1]^2 - u[2] - u[3], u[1], 5.0] .+ 0.1 * randn(6)
        true_u = [3, 1, 2]
        y = G(true_u)
        Γ = (0.1)^2 * I

        prior_u1 = constrained_gaussian("positive_with_mean_1", 2, 1, 0, Inf)
        prior_u2 = constrained_gaussian("two_with_spread_2", 0, 5, -Inf, Inf, repeats = 2)
        prior = combine_distributions([prior_u1, prior_u2])

        N_ensemble = 6
        initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
        ekp = EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(), verbose = true)

        N_iterations = 8
        for i in 1:N_iterations
            params_i = get_ϕ_final(prior, ekp)

            G_matrix = hcat(
                [G(params_i[:, i]) for i in 1:N_ensemble]..., # Parallelize here!
            )

            update_ensemble!(ekp, G_matrix)
        end

        # Test functions for plotting distributions and errors over time or iterations
        fig1 = CairoMakie.Figure(size = (200 * 4, 200 * 7))
        layout = fig1[1, 1:2] = CairoMakie.GridLayout()
        EnsembleKalmanProcesses.Visualize.plot_parameter_distribution(layout, prior)
        EnsembleKalmanProcesses.Visualize.plot_error_over_iters(
            fig1[2, 1],
            ekp,
            color = :tomato,
            axis = (xlabel = "Iterations [added by axis keyword argument]",),
        )
        EnsembleKalmanProcesses.Visualize.plot_error_over_time(
            fig1[2, 2],
            ekp,
            linestyle = :dash,
            auto_log_scale = true,
            error_metric = "bayes_loss",
        )
        ax1 = CairoMakie.Axis(fig1[3, 1], title = "Error over iterations (called from mutating function)")
        ax2 = CairoMakie.Axis(fig1[3, 2], title = "Error over time (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_error_over_iters!(ax1, ekp, color = :aquamarine, auto_log_scale = true)
        EnsembleKalmanProcesses.Visualize.plot_error_over_time!(
            ax2,
            ekp,
            linestyle = :dashdotdot,
            auto_log_scale = true,
            error_metric = "bayes_loss",
        )
        save(joinpath(tmp_dir, "priors_and_errors.png"), fig1)

        # Test functions for plotting ϕ over time or iterations
        fig2 = CairoMakie.Figure(size = (400 * 2, 400 * 2))
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_iters(fig2[1, 1], ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_time(fig2[1, 2], ekp, prior, 2)
        ax1 = CairoMakie.Axis(fig2[2, 1], title = "Constrained parameters (called from mutating function)")
        ax2 = CairoMakie.Axis(fig2[2, 2], title = "Constrained parameters (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_iters!(ax1, ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_time!(ax2, ekp, prior, 2)

        save(joinpath(tmp_dir, "phi.png"), fig2)

        # Test functions for plotting mean ϕ over time or iterations
        fig3 = CairoMakie.Figure(size = (400 * 2, 400 * 2))
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_iters(fig3[1, 1], ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_time(fig3[1, 2], ekp, prior, 2)
        ax1 = CairoMakie.Axis(fig3[2, 1], xlabel = "Iters (called from mutating function)")
        ax2 = CairoMakie.Axis(fig3[2, 2], xlabel = "Time (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_iters!(ax1, ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_time!(ax2, ekp, prior, 2)
        save(joinpath(tmp_dir, "mean_phi.png"), fig3)

        # Test functions for plotting mean and std of ϕ over time or iterations
        fig4 = CairoMakie.Figure(size = (400 * 2, 400 * 2))
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_iters(
            fig4[1, 1],
            ekp,
            prior,
            1,
            plot_std = true,
            band_kwargs = (alpha = 0.2,),
            color = :red,
        )
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_time(
            fig4[1, 2],
            ekp,
            prior,
            2,
            plot_std = true,
            line_kwargs = (linestyle = :dash,),
            band_kwargs = (alpha = 0.2,),
            color = :purple,
        )
        ax1 = CairoMakie.Axis(fig4[2, 1], xlabel = "Iters (called from mutating function)")
        ax2 = CairoMakie.Axis(fig4[2, 2], xlabel = "Time (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_iters!(
            ax1,
            ekp,
            prior,
            1,
            plot_std = true,
            band_kwargs = (color = (:blue, 0.2),),
            color = :red,
        )
        EnsembleKalmanProcesses.Visualize.plot_ϕ_mean_over_time!(
            ax2,
            ekp,
            prior,
            2,
            plot_std = true,
            line_kwargs = (linestyle = :dash,),
            band_kwargs = (color = (:green, 0.2),),
            color = :purple,
        )

        save(joinpath(tmp_dir, "mean_and_std_phi.png"), fig4)

        # Access plots at tmp_dir
        # Print info again since it is a little tedious to find it if you miss it
        @info "Tempdir", tmp_dir
    end
end
