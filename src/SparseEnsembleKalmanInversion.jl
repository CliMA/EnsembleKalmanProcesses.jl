#Sparse Ensemble Kalman Inversion: specific structures and function definitions
using Convex, SCS
using SparseArrays

"""
    SparseInversion <: Process

A sparse ensemble Kalman Inversion process
"""
struct SparseInversion{FT <: AbstractFloat, IT <: Int} <: Process
    "upper limit of l1-norm"
    γ::FT
    "the flag to enable thresholding"
    threshold_eki::Bool
    "a small value to threshold results that are close to zero"
    threshold_value::FT
    "a small value for regularization to enhance robustness of convex optimization"
    reg::FT
    "a list of index to indicate the parameters included in the evaluation of l1-norm"
    uc_idx::AbstractVector{IT}
end

"""
    sparse_qp(
        ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT,IT}},
        v_j::Vector{FT},
        cov_vv_inv::AbstractMatrix{FT},
        H_u::SparseArrays.SparseMatrixCSC{FT},
        H_g::SparseArrays.SparseMatrixCSC{FT},
        y_j::Vector{FT};
        H_uc::SparseArrays.SparseMatrixCSC{FT} = H_u,
    ) where {FT, IT}

Solving quadratic programming problem with sparsity constraint.
"""
function sparse_qp(
    ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT, IT}},
    v_j::Vector{FT},
    cov_vv_inv::AbstractMatrix{FT},
    H_u::AbstractMatrix{FT},
    H_g::AbstractMatrix{FT},
    y_j::Vector{FT};
    H_uc::AbstractMatrix{FT} = H_u,
) where {FT, IT}

    P = H_g' * (ekp.obs_noise_cov \ H_g) + cov_vv_inv
    P = 0.5 * (P + P')
    q = -(cov_vv_inv * v_j + H_g' * (ekp.obs_noise_cov \ y_j))
    N_params = size(H_uc)[1]
    P1 = vcat(
        hcat(P, fill(FT(0), size(P)[1], N_params)),
        hcat(fill(FT(0), N_params, size(P)[1]), fill(FT(0), N_params, N_params)),
    )
    q1 = vcat(q, fill(FT(0), N_params, 1))
    H_uc_abs = 1.0 * I(N_params)
    G = hcat(vcat(H_uc, -1.0 * H_uc), vcat(-1.0 * H_uc_abs, -1.0 * H_uc_abs))
    h = fill(FT(0), 2 * N_params, 1)
    G1 = vcat(G, hcat(fill(FT(0), 1, size(P)[1]), fill(FT(1), 1, N_params)))
    h1 = vcat(h, ekp.process.γ)
    x = Variable(size(P1)[1])
    problem = minimize(0.5 * quadform(x, P1; assume_psd = true) + q1' * x, [G1 * x <= h1])
    solve!(problem, () -> SCS.Optimizer(verbose = false))

    return hcat(H_u, fill(FT(0), size(H_u)[1], N_params)) * evaluate(x)
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, <:SparseInversion{FT,IT}},
        g::Array{FT,2};
        cov_threshold::FT=0.01,
        Δt_new=nothing
    ) where {FT, IT}

Updates the ensemble according to which type of Process we have. Model outputs `g` need to be a `N_obs × N_ens` array (i.e data are columms).
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT, IT}},
    g::AbstractMatrix{FT};
    cov_threshold::FT = 0.01,
    Δt_new = nothing,
    deterministic_forward_map = true,
) where {FT, IT}

    # Update follows eqns. (4) and (5) of Schillings and Stuart (2017)

    #catch works when g non-square
    if !(size(g)[2] == ekp.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g do not match, try transposing g or check ensemble size"))
    end

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)
    N_obs = size(g, 1)

    cov_init = cov(u, dims = 2)
    cov_ug = cov(u, g, dims = 2, corrected = false) # [N_par × N_obs]
    cov_gg = cov(g, g, dims = 2, corrected = false) # [N_obs × N_obs]
    cov_uu = cov(u, u, dims = 2, corrected = false) # [N_par × N_par]

    set_Δt!(ekp, Δt_new)

    v = hcat(u', g')

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(ekp.rng, MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = (cov_gg + scaled_obs_noise_cov) \ (y - g)
    u += (cov_ug * tmp) # [N_par × N_ens]

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns = true))
    push!(ekp.g, DataContainer(g, data_are_columns = true))

    # Sparse EKI
    cov_vv = vcat(hcat(cov_uu, cov_ug), hcat(cov_ug', cov_gg))
    H_u = hcat(1.0 * I(size(u)[1]), fill(FT(0), size(u)[1], size(g)[1]))
    H_g = hcat(fill(FT(0), size(g)[1], size(u)[1]), 1.0 * I(size(g)[1]))

    H_uc = H_u[ekp.process.uc_idx, :]

    cov_vv_inv = cov_vv \ (1.0 * I(size(cov_vv)[1]))

    # Loop over ensemble members to impose sparsity
    Threads.@threads for j in 1:(ekp.N_ens)
        # Solve a quadratic programming problem
        u[:, j] = sparse_qp(ekp, v[j, :], cov_vv_inv, H_u, H_g, y[:, j], H_uc = H_uc)

        # Threshold the results if needed
        if ekp.process.threshold_eki
            u[ekp.process.uc_idx, j] =
                u[ekp.process.uc_idx, j] .* (abs.(u[ekp.process.uc_idx, j]) .> ekp.process.threshold_value)
        end

        # Add small noise to constrained elements of u
        u[ekp.process.uc_idx, j] += rand(
            ekp.rng,
            MvNormal(zeros(size(ekp.process.uc_idx)[1]), ekp.process.reg * I(size(ekp.process.uc_idx)[1])),
        )
    end

    # Store error
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims = 2)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string(
            "New ensemble covariance determinant is less than ",
            cov_threshold,
            " times its previous value.",
            "\nConsider reducing the EK time step.",
        )
    end
end
