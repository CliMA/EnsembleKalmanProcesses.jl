#Ensemble Kalman Inversion: specific structures and function definitions
using CVXOPT
using SparseArrays

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

"""
    find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}

Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::Array{FT, 2};
    cov_threshold::FT = 0.01,
) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims = 2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new = Δt)
        cov_new = cov(get_u_final(ekp_copy), dims = 2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    return Δt

end

"""
    sparse_qp(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, v_j::Vector{FT}, cov_vv_inv::Array{FT, 2}, H_u::SparseArrays.SparseMatrixCSC{FT}, H_g::SparseArrays.SparseMatrixCSC{FT}, y_j::Vector{FT}, γ::FT; H_uc::SparseArrays.SparseMatrixCSC{FT} = H_u) where {FT, IT} 

Solving quadratic programming problem with sparsity constraint.
"""
function sparse_qp(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion}, 
    v_j::Vector{FT}, 
    cov_vv_inv::Array{FT, 2}, 
    H_u::SparseArrays.SparseMatrixCSC{FT}, 
    H_g::SparseArrays.SparseMatrixCSC{FT}, 
    y_j::Vector{FT}, 
    γ::FT; 
    H_uc::SparseArrays.SparseMatrixCSC{FT} = H_u
) where {FT, IT}

    P = H_g' * inv(ekp.obs_noise_cov) * H_g + cov_vv_inv
    P = 0.5 * (P + P')
    q = -(cov_vv_inv*v_j + H_g' * inv(ekp.obs_noise_cov) * y_j)
    N_params = size(H_uc)[1]
    P1 = vcat(hcat(P, fill(FT(0), size(P)[1], N_params)), 
              hcat(fill(FT(0), N_params, size(P)[1]), fill(FT(0), N_params, N_params)))
    q1 = vcat(q, fill(FT(0), N_params, 1))
    H_uc_abs = 1.0 * I(N_params)
    G = hcat(vcat(H_uc, -1.0 * H_uc), vcat(-1.0 * H_uc_abs, -1.0 * H_uc_abs))
    h = fill(FT(0), 2*N_params, 1)
    G1 = vcat(G, hcat(fill(FT(0), 1, size(P)[1]), fill(FT(1), 1, N_params)))
    h1 = vcat(h, γ)
    options = Dict([("show_progress",false)])
    sol = CVXOPT.qp(P1, q1, G1, h1, options=options)

    return hcat(H_u, fill(FT(0), size(H_u)[1], N_params)) * sol["x"]
end

"""
    update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, <:Inversion}, g::Array{FT,2} cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}

Updates the ensemble according to which type of Process we have. Model outputs `g` need to be a `N_obs × N_ens` array (i.e data are columms).
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::Array{FT, 2};
    cov_threshold::FT = 0.01,
    Δt_new = nothing,
    deterministic_forward_map = true,
    sparse_eki = false,
    γ = 10,
    threshold_eki = false,
    threshold_value = 1e-2,
    reg = 1e-6,
    uc_idx = []
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

    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end

    v = hcat(u', g')

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

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

    if sparse_eki
        # Sparse EKI
        cov_vv = vcat(hcat(cov_uu, cov_ug), hcat(cov_ug',cov_gg))
        H_u = hcat(1.0 * I(size(u)[1]), fill(FT(0), size(u)[1], size(g)[1]))
        H_g = hcat(fill(FT(0), size(g)[1], size(u)[1]), 1.0 * I(size(g)[1]))
        
        if uc_idx == []
            H_uc = H_u
        else
            H_uc = H_u[uc_idx,:]
        end

        for j = 1:ekp.N_ens
            # Solve a quadratic programming problem
            u[:,j] = sparse_qp(ekp, v[j,:], inv(cov_vv), H_u, H_g, y[:,j], γ, H_uc=H_uc)

            # Threshold the results if needed
            if threshold_eki
                if uc_idx == []
                    u[:,j] = u[:,j] .* (abs.(u[:,j]) .> threshold_value)
                else
                    u[uc_idx,j] = u[uc_idx,j] .* (abs.(u[uc_idx,j]) .> threshold_value)
                end
            end

            # Add small noise to constrained elements of u
            if uc_idx == []
                u[:,j] += rand(MvNormal(zeros(size(u)[1]), reg * I(size(u)[1])))
            else
                u[uc_idx,j] += rand(MvNormal(zeros(size(uc_idx)[1]), reg * I(size(uc_idx)[1])))
            end
        end
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
