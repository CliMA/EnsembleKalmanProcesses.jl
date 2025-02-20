# Ensemble Kalman Sampler (Nonreversible): specific structures and function definitions

"""
    NonreversibleSampler{FT<:AbstractFloat,IT<:Int} <: Process

An ensemble Kalman Sampler with (non-reversible preconditiong) process.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct NonreversibleSampler{FT <: AbstractFloat} <: Process
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::Vector{FT}
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}}
    "prefactor - giving the sqrt of condition number of J_opt"
    prefactor::FT
end

get_prefactor(ns::NonreversibleSampler) = ns.prefactor


function NonreversibleSampler(prior::ParameterDistribution; prefactor = 1.1)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    return NonreversibleSampler{FT}(mean_prior, cov_prior, FT(prefactor))
end


function FailureHandler(process::NonreversibleSampler, method::IgnoreFailures)
    function failsafe_update(ekp, u, g, failed_ens)
        u_transposed = permutedims(u, (2, 1))
        g_transposed = permutedims(g, (2, 1))
        u_transposed = eksnr_update(ekp, u_transposed, g_transposed)
        u_new = permutedims(u_transposed, (2, 1))
        return u_new
    end
    return FailureHandler{NonreversibleSampler, IgnoreFailures}(failsafe_update)
end



"""
     eksnr_update(
        ekp::EnsembleKalmanProcess{FT, IT, NonreversibleSampler{FT}},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
    ) where {FT <: Real, IT}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the sampler algorithm.

The current implementation assumes that rows of u and g correspond to
ensemble members, so it requires passing the transpose of the `u` and
`g` arrays associated with ekp.
"""
function eksnr_update(
    ekp::EnsembleKalmanProcess{FT, IT, NonreversibleSampler{FT}},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
) where {FT <: Real, IT}
    # TODO: Work with input data as columns

    tol = 1e8 * eps()

    # u_mean: N_par × 1
    u_mean = mean(u', dims = 2)
    dimen = size(u_mean, 1)
    #=if dimen != 2
        throw(ArgumentError("Nonreversible Implementation is only for dimension = 2, received $dimen"))
    end
=#
    
    # g_mean: N_obs × 1
    u_mean = mean(u', dims = 2)
    g_mean = mean(g', dims = 2)
    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov([u'; g'], dims = 2, corrected = false), dimen)

    # Build the preconditioners:
    decomp_Cuu = eigen(cov_uu)
    # .values = min -> max evals,
    # .vectors stores evecs as rows
    # D̃_opt = d * λ_min⁻¹ * v_min ⊗ v_min = λ_min⁻¹ D_opt
    λ_min = decomp_Cuu.values[1]
    v = decomp_Cuu.vectors[:, 1]
    
    # normalize
    v /= norm(v) # normalize v
    λ_min = norm(v) * λ_min #rescale lambda for a normlized 
    λ_opt = 1 / λ_min
    D_opt = Symmetric(dimen * v * v')
    D̃_opt = λ_opt * D_opt

    # orthonormal basis such that ⟨Ψₖ, D̃_optΨₖ⟩ = Tr(D̃_opt) / d
    # Assume first: e_k are the standard basis
    ξ = 1.0 / sqrt(dimen) .* ones(dimen) # true when e_k standard basis

    # create ṽ, the (normalized) normal vector from v that passes through ξ
    ṽ = ξ - dot(ξ, v) * v
    ṽ /= norm(ṽ)

    θ = acos(dot(ξ, v))
    T1 = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    T2 = [cos(-θ) -sin(-θ); sin(-θ) cos(-θ)]
    A_θ = all(abs.(T1 * [dot(ξ, v); dot(ξ, ṽ)] .- [1.0; 0.0]) .< tol) ? T1 : T2
    @assert all(abs.(A_θ * [dot(ξ, v); dot(ξ, ṽ)] .- [1.0; 0.0]) .< tol) # just make sure T2 version works too...

    Ψ = zeros(dimen, dimen) #columns are evecs
    for k in 1:dimen
        # true when e_k standard basis 
        e_parr_coeff = [v[k] ṽ[k]]
        e_parr_vec = [v ṽ]'

        if dimen > 2
            e_parr = e_parr_coeff * e_parr_vec
            # e_perp satisfies e_parr + e_perp = e_k
            e_perp = -e_parr
            e_perp[k] += 1.0

            Ψ[k, :] = (A_θ * e_parr_coeff')' * e_parr_vec + e_perp
        else
            Ψ[k, :] = (A_θ * e_parr_coeff')' * e_parr_vec
        end
        Ψ[k, :] /= norm(Ψ[k, :])
    end
    # useful test: if this produces a ONB

    for j in 1:dimen
        for k in 1:dimen
            if j == k
                @assert (norm(Ψ[k, :]) - 1.0 < tol) #true if normal
            else
                @assert (abs(dot(Ψ[k, :], Ψ[j, :])) < tol) #true if orthogonal
            end
        end
    end
   
    @assert all(abs.([norm(Ψ[k,:]) for k=1:dimen] .- ones(dimen)) .< tol) #true if normal

    # now calculate J
    process = get_process(ekp)
    prefactor = get_prefactor(process)
    if prefactor <= 1
        throw(ArgumentError("Nonreversible prefactor must be > 1, continuing with value 1.1"))
    end
    
    λ = [(dimen - 1) / (prefactor^2 - 1) + k - 1 for k in 1:dimen]
    Ĵ_opt = zeros(dimen, dimen)
    for k in 1:(dimen - 1)
        for j in (k + 1):dimen
#            Ĵ_opt[j, k] = (λ[j] + λ[k]) / (λ[j] - λ[k]) * (dimen / λ_min) * dot(Ψ[j, :], v) * dot(Ψ[k, :], v)
            Ĵ_opt[j, k] = (λ[j] + λ[k]) / (λ[j] - λ[k]) * (1 / λ_min) 
            Ĵ_opt[k, j] = -Ĵ_opt[j, k]
#            @assert  isapprox(dot(Ψ[j, :], v) * dot(Ψ[k, :], v),1 ./ dimen) # 
        end
    end
    #@info "λs = $(λ)"
    
    ### !!! note that for Lin alg below the Psi basis vectors should be columns
    Ψ=Ψ'
    ### 
    
    sqC = sqrt(cov_uu)
    # as ΨΨ' = I
    J_opt = sqC * Ψ * Ĵ_opt * Ψ' * sqC
    
    # (D̃_opt + JJ_opt) * Q + Q * (D̃_opt - JJ_opt) = 2 * λ_opt * Q
    # Sanity check: Lyapunov condition from Arnold, Signorello 2021 eqn (3.7)
    JJ_opt = Ψ * Ĵ_opt * Ψ'
    Q = Ψ * Diagonal(λ) * Ψ'
    Qinv = inv(Q)
    @assert all(isapprox.(Qinv*((JJ_opt + D̃_opt)*Q + Q*(D̃_opt-JJ_opt)) ./(2*λ_opt), I(dimen); atol=1e-8 ))

    # Default: Δt = 1 / (norm(grad(misfit)) + eps(FT))
    Δt = get_Δt(ekp)[end]
    noise_cov = MvNormal(zeros(dimen), I(dimen)) # only D in here
    obs_noise_cov_inv = get_obs_noise_cov_inv(ekp)

    # uncomment to see EKS behaviour (check timestepping too)
    #= 
    D_opt = cov_uu
    J_opt = zeros(size(J_opt))
    =#
    
    # From Alg 3: Explicit scheme!
    # u_n+1 = u_n - Δt(D+J)[(C^uu)^-1 C^ug Γ⁻¹(y - g') + C_0⁻¹ (u' - m_0)] + sqrt(2Δt)*χ,  χ∼N(0,D)    
    #=
        lhs =
            u' -
            Δt *
            (D_opt + J_opt) *
            (
                cov_uu \ (cov_ug * obs_noise_cov_inv * (get_obs(ekp) .- g')) -
                process.prior_cov \ (process.prior_mean .- u' )
            )

        u = lhs' + sqrt(2 * Δt) * (sqrt(D_opt) * rand(get_rng(ekp), noise_cov, get_N_ens(ekp)))'
    =#

    # Alg 3: Split-implicit update
    # u*_n+1 = u_n - Δt(D+J)( (C^uu)^-1 C^ug Γ⁻¹(y - g') + C_0⁻¹ (u*_{n+1}' - m_0)] + sqrt(2Δt)*χ,  χ∼N(0,D)
    implicit =
        (I(size(u, 2)) + Δt * (process.prior_cov' \ (D_opt + J_opt)')') \ (
            u' .-
            Δt *
            (D_opt + J_opt) *
            (
                (cov_uu \ cov_ug * obs_noise_cov_inv * (g' .- get_obs(ekp))) .-
                (process.prior_cov \ process.prior_mean)
            )
        )
    u = implicit' + sqrt(2 * Δt) * (sqrt(D_opt) * rand(get_rng(ekp), noise_cov, get_N_ens(ekp)))'
    @info "λ_opt: $(λ_opt)"

#=
    # quantities of interest
    # lambda_opt*(d+sqrt(cond(C))*(2*pi*c^2)/(sqrt(3)(c^2-1))*sqrt(d)(d-1)  
    ubd_paper = λ_opt * (dimen + sqrt(cond(cov_uu)) * 2*pi*prefactor^2/(sqrt(3)*(prefactor^2-1))*sqrt(dimen)*(dimen-1))
    DpJCinv = (D_opt + J_opt)*inv(cov_uu)
    ubd = norm(DpJCinv)
    @info "$(ubd) < $(ubd_paper)"
    eval_ubd = eigen(DpJCinv)
    @info "$(eval_ubd.values)"
    @info "$(cond(DpJCinv))"
=#
    
    return u
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, NonreversibleSampler{FT}},
        g::AbstractMatrix{FT},
        process::NonreversibleSampler{FT};
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to a NonreversibleSampler process. 

Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - process :: Type of the EKP.
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, NonreversibleSampler{FT}},
    g::AbstractMatrix{FT},
    process::NonreversibleSampler{FT},
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
) where {FT, IT}

    # u: N_ens × N_par
    # g: N_ens × N_obs
    u_old = get_u_final(ekp)
    cov_init = get_u_cov_final(ekp)

    fh = ekp.failure_handler

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u_old, g, failed_ens)

    return u
end
