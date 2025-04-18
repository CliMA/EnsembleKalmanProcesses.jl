if TEST_PLOT_OUTPUT
    using Test
    using CairoMakie

    using LinearAlgebra
    using EnsembleKalmanProcesses
    using EnsembleKalmanProcesses.ParameterDistributions

    @testset "Makie ploting" begin
        tmp_dir = mktempdir(cleanup = false)
        # Access plots at tmp_dir
        @info "Tempdir", tmp_dir

        G(u) = [1 / abs(u[1]), sum(u[2:5]), prod(u[3:4]), u[1]^2 - u[2] - u[3], u[4], u[5]^3] .+ 0.1 * randn(6)
        true_u = [3, 1, 2, -3, -4]
        y = G(true_u)
        Γ = (0.1)^2 * I

        prior_u1 = constrained_gaussian("positive_with_mean_2", 2, 1, 0, Inf)
        prior_u2 = constrained_gaussian("four_with_spread_5", 0, 5, -Inf, Inf, repeats = 4)
        prior = combine_distributions([prior_u1, prior_u2])

        N_ensemble = 5
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

        # Test functions for plotting priors and errors over time or iterations
        fig = CairoMakie.Figure(size = (200 * 4, 200 * 7))
        layout = fig[1, 1:2] = CairoMakie.GridLayout()
        EnsembleKalmanProcesses.Visualize.plot_priors(layout, prior)
        EnsembleKalmanProcesses.Visualize.plot_error_over_iters(
            fig[2, 1],
            ekp,
            color = :tomato,
            axis = (xlabel = "Iterations [added by axis keyword argument]",),
        )
        EnsembleKalmanProcesses.Visualize.plot_error_over_time(fig[2, 2], ekp, linestyle = :dash)
        ax1 = CairoMakie.Axis(fig[3, 1], title = "Error over iterations (called from mutating function)")
        ax2 = CairoMakie.Axis(fig[3, 2], title = "Error over time (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_error_over_iters!(ax1, ekp, color = :aquamarine)
        EnsembleKalmanProcesses.Visualize.plot_error_over_time!(ax2, ekp, linestyle = :dashdotdot)
        save(joinpath(tmp_dir, "priors_and_errors.png"), fig)

        # Test functions for plotting ϕ over time or iterations
        fig = CairoMakie.Figure(size = (200 * 4, 200 * 4))
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_iters(fig[1, 1], ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_time(fig[1, 2], ekp, prior, 2)
        ax1 = CairoMakie.Axis(fig[2, 1], title = "Constrained parameters (called from mutating function)")
        ax2 = CairoMakie.Axis(fig[2, 2], title = "Constrained parameters (called from mutating function)")
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_iters!(ax1, ekp, prior, 1)
        EnsembleKalmanProcesses.Visualize.plot_ϕ_over_time!(ax2, ekp, prior, 2)

        save(joinpath(tmp_dir, "phi.png"), fig)

        # Access plots at tmp_dir
        # Print info again since it is a little tedious to find it the first time
        @info "Tempdir", tmp_dir
    end
end
