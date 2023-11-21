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
    "prefactor"
    prefactor::FT
end

get_prefactor(ns::NonreversibleSampler) = ns.prefactor


function NonreversibleSampler(prior::ParameterDistribution; prefactor = 1.1)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    return NonreversibleSampler{FT}(mean_prior, cov_prior, prefactor)
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
    if dimen != 2
        throw(ArgumentError("Nonreversible Implementation is only for dimension = 2, received $dimen"))
    end

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
    λ_min = norm(v) * λ_min #rescale lambda for a normlized v

    D_opt = dimen * v * v'
    #D̃_opt = 1 / λ_min * D_opt

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


    #@assert all(abs.([norm(Ψ[k,:]) for k=1:dimen] .- ones(dimen)) .< tol) #true if normal

    # now calculate J
    prefactor = get_prefactor(ekp.process)
    if prefactor <= 1
        throw(ArgumentError("Nonreversible prefactor must be > 1, continuing with value 1.1"))
    end

    λ = [(dimen - 1) / (prefactor^2 - 1) + k - 1 for k in 1:dimen]
    Ĵ_opt = zeros(dimen, dimen)
    for k in 1:(dimen - 1)
        for j in (k + 1):dimen

            #            Ĵ_opt[j, k] = (λ[j] + λ[k]) / (λ[j] - λ[k]) * Ψ[j, :]' * D̃_opt * Ψ[k, :]
            Ĵ_opt[j, k] = (λ[j] + λ[k]) / (λ[j] - λ[k]) * (dimen / λ_min) * dot(Ψ[j, :], v) * dot(Ψ[k, :], v)
            Ĵ_opt[k, j] = -Ĵ_opt[j, k]

        end
    end
    sqC = sqrt(cov_uu)
    # as ΨΨ' = I
    J_opt = sqC * Ψ * Ĵ_opt * Ψ' * sqC

    # Default: Δt = 1 / (norm(D) + eps(FT))
    Δt = ekp.Δt[end]
    noise_cov = MvNormal(zeros(dimen), I(dimen)) # only D in here


    # From Alg 3: Explicit scheme!
    # u_n+1 = u_n - Δt(D+J)[(C^uu)^-1 C^ug Γ⁻¹(y - g') + C_0⁻¹ (m_0 - u')] + sqrt(2Δt)*χ,  χ∼N(0,D)    
    #=
        implicit =
            u' -
            Δt *
            (D_opt + J_opt) *
            (
                cov_uu \ (cov_ug * (ekp.obs_noise_cov \ (ekp.obs_mean .- g'))) +
                ekp.process.prior_cov \ (ekp.process.prior_mean .- u')
            )

        u = implicit' + sqrt(2 * Δt) * (sqrt(D_opt) * rand(ekp.rng, noise_cov, ekp.N_ens))'
    =#

    # Alg 3: Split-implicit update
    implicit =
        (I(size(u, 2)) + Δt * (ekp.process.prior_cov' \ (D_opt + J_opt)')') \ (
            u' .-
            Δt *
            (D_opt + J_opt) *
            (
                (cov_uu \ (cov_ug * (ekp.obs_noise_cov \ (g' .- ekp.obs_mean)))) .+
                (ekp.process.prior_cov \ ekp.process.prior_mean)
            )
        )

    u = implicit' + sqrt(2 * Δt) * (sqrt(D_opt) * rand(ekp.rng, noise_cov, ekp.N_ens))'

    # (rewritten update in style of EKS: same as split-implicit)
    #=    
        E = g' .- g_mean
        R = g' .- ekp.obs_mean
        # D: N_ens × N_ens
        F = (1 / ekp.N_ens) * (E' * (ekp.obs_noise_cov \ R))

        implicit =
            (I(size(u,2)) + Δt * (ekp.process.prior_cov' \ (D_opt + J_opt)')') \
            (u' .- Δt * (D_opt + J_opt) * ((cov_uu \ (u' .- u_mean) * F) .+ (ekp.process.prior_cov \ ekp.process.prior_mean)))       

        u = implicit' + sqrt(2 * Δt) * (sqrt(D_opt) * rand(ekp.rng, noise_cov, ekp.N_ens))'
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
    process::NonreversibleSampler{FT};
    failed_ens = nothing,
) where {FT, IT}

    # u: N_ens × N_par
    # g: N_ens × N_obs
    u_old = get_u_final(ekp)
    cov_init = get_u_cov_final(ekp)

    fh = ekp.failure_handler

    if ekp.verbose
        if get_N_iterations(ekp) == 0
            @info "Iteration 0 (prior)"
            @info "Covariance trace: $(tr(cov_init))"
        end

        @info "Iteration $(get_N_iterations(ekp)+1) (T=$(sum(ekp.Δt)))"
    end

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u = fh.failsafe_update(ekp, u_old, g, failed_ens)

    # store new parameters (and model outputs)
    push!(ekp.g, DataContainer(g, data_are_columns = true))
    # u_old is N_ens × N_par, g is N_ens × N_obs,
    # but stored in data container with N_ens as the 2nd dim

    compute_error!(ekp)

    # Diagnostics
    cov_new = cov(u, dims = 2)

    if ekp.verbose
        @info "Covariance-weighted error: $(get_error(ekp)[end])\nCovariance trace: $(tr(cov_new))\nCovariance trace ratio (current/previous): $(tr(cov_new)/tr(cov_init))"
    end

    return u
end
