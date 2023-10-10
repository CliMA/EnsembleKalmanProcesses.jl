#Sparse Ensemble Kalman Inversion: specific structures and function definitions
using Convex, SCS, MathOptInterface
using SparseArrays

const MOI = MathOptInterface

"""
    SparseInversion <: Process

A sparse ensemble Kalman Inversion process

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
Base.@kwdef struct SparseInversion{FT <: AbstractFloat} <: Process
    "upper limit of l1-norm"
    γ::FT
    "threshold below which the norm of parameters is pruned to zero"
    threshold_value::FT = FT(0)
    "indices of parameters included in the evaluation of l1-norm constraint"
    uc_idx::Union{AbstractVector, Colon} = Colon()
    "a small regularization value to enhance robustness of convex optimization"
    reg::FT = FT(0)
end

function SparseInversion(
    γ::FT;
    threshold_value::FT = FT(0),
    uc_idx::Union{AbstractVector, Colon} = Colon(),
    reg::FT = FT(0),
) where {FT <: AbstractFloat}
    return SparseInversion{FT}(γ, threshold_value, uc_idx, reg)
end

function FailureHandler(process::SparseInversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens) = sparse_eki_update(ekp, u, g, y, obs_noise_cov)
    return FailureHandler{SparseInversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::SparseInversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the SparseEKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::SparseInversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            sparse_eki_update(ekp, u[:, successful_ens], g[:, successful_ens], y[:, successful_ens], obs_noise_cov)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{SparseInversion, SampleSuccGauss}(failsafe_update)
end

"""
    sparse_qp(
        ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
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
    ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
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
    solve!(problem, MOI.OptimizerWithAttributes(SCS.Optimizer, "verbose" => 0))

    return hcat(H_u, fill(FT(0), size(H_u)[1], N_params)) * evaluate(x)
end

"""
     sparse_eki_update(
        ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
        y::AbstractMatrix{FT},
        obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
    ) where {FT <: Real, CT <: Real, IT}

Returns the sparse updated parameter vectors given their current values and
the corresponding forward model evaluations, using the inversion algorithm
from eqns. (3.7) and (3.14) of Schneider et al. (2021).

Localization is applied following Tong and Morzfeld (2022).
"""
function sparse_eki_update(
    ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    y::AbstractMatrix{FT},
    obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
) where {FT <: Real, CT <: Real, IT}

    cov_est = cov([u; g], [u; g], dims = 2, corrected = false) # [(N_par + N_obs)×(N_par + N_obs)]

    # Localization
    cov_localized = ekp.localizer.localize(cov_est)
    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov_localized, size(u, 1))

    v = hcat(u', g')

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = FT.((cov_gg + obs_noise_cov) \ (y - g))
    u = u + (cov_ug * tmp) # [N_par × N_ens]

    # Sparse EKI
    cov_vv = vcat(hcat(cov_uu, cov_ug), hcat(cov_ug', cov_gg))
    H_u = hcat(1.0 * I(size(u)[1]), fill(FT(0), size(u)[1], size(g)[1]))
    H_g = hcat(fill(FT(0), size(g)[1], size(u)[1]), 1.0 * I(size(g)[1]))

    H_uc = H_u[ekp.process.uc_idx, :]

    cov_vv_inv = cov_vv \ (1.0 * I(size(cov_vv)[1]))

    # Loop over ensemble members to impose sparsity
    Threads.@threads for j in 1:size(u, 2)
        # Solve a quadratic programming problem
        u[:, j] = sparse_qp(ekp, v[j, :], cov_vv_inv, H_u, H_g, y[:, j], H_uc = H_uc)

        # Prune parameters using threshold
        u[ekp.process.uc_idx, j] =
            u[ekp.process.uc_idx, j] .* (abs.(u[ekp.process.uc_idx, j]) .> ekp.process.threshold_value)

        # Add small noise to constrained elements of u
        if isposdef(ekp.process.reg)
            len_u_sparse = length(u[ekp.process.uc_idx, j])
            u[ekp.process.uc_idx, j] += rand(ekp.rng, MvNormal(zeros(len_u_sparse), ekp.process.reg * I(len_u_sparse)))
        end
    end
    return u
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
        g::AbstractMatrix{FT},
        process::SparseInversion{FT};
        deterministic_forward_map = true,
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to a SparseInversion process. 

Inputs:
 - `ekp` :: The EnsembleKalmanProcess to update.
 - `g` :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - `process` :: Type of the EKP.
 - `deterministic_forward_map` :: Whether output `g` comes from a deterministic model.
 - `failed_ens` :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, SparseInversion{FT}},
    g::AbstractMatrix{FT},
    process::SparseInversion{FT};
    deterministic_forward_map = true,
    failed_ens = nothing,
) where {FT, IT}

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)
    N_obs = size(g, 1)
    cov_init = cov(u, dims = 2)
    fh = ekp.failure_handler

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(ekp.rng, MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u, g, y, scaled_obs_noise_cov, failed_ens)

    # store new parameters (and model outputs)
    push!(ekp.g, DataContainer(g, data_are_columns = true))

    # Store error
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims = 2)

    return u
end
