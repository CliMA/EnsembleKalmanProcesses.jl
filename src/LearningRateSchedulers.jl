# included in EnsembleKalmanProcess.jl

export DefaultScheduler, MutableScheduler, EKSStableScheduler, DataMisfitController
export calculate_timestep!, posdef_correct

# default unless user overrides

"""
$(TYPEDEF)

Scheduler containing a default constant step size,
users can override this temporarily within `update_ensemble!`.

$(TYPEDFIELDS)
"""
struct DefaultScheduler{FT} <: LearningRateScheduler where {FT <: AbstractFloat}
    "step size"
    Δt_default::FT
end

"""
$(TYPEDSIGNATURES)

Sets `Δt_default = 1.0`
"""
function DefaultScheduler()
    return DefaultScheduler{Float64}(Float64(1))
end

function DefaultScheduler(it::IT) where {IT <: Int}
    return DefaultScheduler{Float64}(Float64(it))
end

# takes latest value user has defined
"""
$(TYPEDEF)

Scheduler containing a mutable constant step size,
users can override this permanently within `update_ensemble!`.

$(TYPEDFIELDS)
"""
struct MutableScheduler{FT} <: LearningRateScheduler where {FT <: AbstractFloat}
    "mutable step size"
    Δt_mutable::Vector{FT}
end

function MutableScheduler(ft::R) where {R <: Real}
    if !(R <: AbstractFloat)
        return MutableScheduler{Float64}(Float64[ft])
    else
        return MutableScheduler{R}([ft])
    end
end

"""
$(TYPEDSIGNATURES)

Sets `Δt_mutable = [1.0]`
"""
function MutableScheduler()
    return MutableScheduler{Float64}(Float64[1])
end


# Copied from the EnsembleKalmanSampler.jl src
"""
$(TYPEDEF)

Scheduler known to be stable for EKS,
In particular, ``\\Delta t = \\frac{\\alpha}{\\|U\\| + \\varepsilon}`` where ``U = (G(u) - \\bar{G(u)})^T\\Gamma^{-1}(G(u) - y)``. 
Cannot be overriden.

$(TYPEDFIELDS)
"""
struct EKSStableScheduler{FT} <: LearningRateScheduler where {FT <: AbstractFloat}
    "the numerator ``\\alpha``"
    numerator::FT
    "the nugget term ``\\varepsilon``"
    nugget::FT
end

function EKSStableScheduler(numerator::R, nugget::RR) where {R <: Real, RR <: Real}
    if !(R <: AbstractFloat) || !(RR <: AbstractFloat)
        FT = Float64
        return EKSStableScheduler{FT}(FT(numerator), FT(nugget))
    elseif R != RR
        FT = Float64
        return EKSStableScheduler{FT}(FT(numerator), FT(nugget))
    else
        return EKSStableScheduler{R}(R(numerator), R(nugget))
    end

end

"""
$(TYPEDSIGNATURES)

Sets `numerator = 1.0` and `nugget = eps()`
"""
function EKSStableScheduler()
    return EKSStableScheduler{Float64}(Float64(1), Float64(eps()))
end

"""
$(TYPEDEF)

Scheduler from Iglesias, Yang, 2021, Based on Bayesian Tempering.
Terminates at `T=1` by default, and at this time, ensemble spread provides a (more) meaningful approximation of posterior uncertainty
In particular, for parameters ``\\theta_j`` at step ``n``, to calculate the next timestep
``\\Delta t_n = \\min\\left(\\max\\left(\\frac{J}{2\\Phi}, \\sqrt{\\frac{J}{2\\langle \\Phi, \\Phi \\rangle}}\\right), 1-\\sum^{n-1}_i t_i\\right) `` where ``\\Phi_j = \\|\\Gamma^{-\\frac{1}{2}}(G(\\theta_j) - y)\\|^2``. 
Cannot be overriden by user provided timesteps.
By default termination returns `true` from `update_ensemble!` and 
- if `on_terminate == "stop"`, stops further iteration.
- if `on_terminate == "continue_fixed"`, continues iteration with the final timestep fixed
- if `on_terminate == "continue"`, continues the algorithm (though no longer compares to ``1-\\sum^{n-1}_i t_i``) 
The user may also change the `T` with `terminate_at` keyword.

$(TYPEDFIELDS)
"""
struct DataMisfitController{FT, M, S} <:
       LearningRateScheduler where {FT <: AbstractFloat, M <: AbstractMatrix, S <: AbstractString}
    "the current iteration"
    iteration::Vector{Int}
    "the inverse square-root of the noise covariance is stored"
    inv_sqrt_noise::Vector{M}
    "the algorithm time for termination, default: 1.0"
    terminate_at::FT
    "the action on termination, default: \"stop\", "
    on_terminate::S
end # Iglesias Yan 2021

"""
$(TYPEDSIGNATURES)

Sets `terminate_at = 1.0` and `on_terminate="stop"`
"""
function DataMisfitController(; terminate_at = 1.0, on_terminate = "stop")
    FT = Float64
    M = Matrix{FT}
    iteration = Int[]
    inv_sqrt_noise = M[]

    if terminate_at > 0 #can be infinity
        ta = FT(terminate_at)
    else
        ta = FT(1.0) # has a notion of posterior
    end

    if on_terminate ∉ ["continue", "continue_fixed", "stop"]
        throw(
            ArgumentError(
                "Unknown keyword option for `on_terminate`, expected \"continue\", \"continue_fixed\", or \"stop\". User provided $(on_terminate)",
            ),
        )
    end

    return DataMisfitController{FT, M, typeof(on_terminate)}(iteration, inv_sqrt_noise, ta, on_terminate)
end

"""
$(TYPEDSIGNATURES)

Calculates next timestep by pushing to ekp.Δt, 
`!isnothing(return_value)` implies termination condition has been met
"""
function calculate_timestep!(
    ekp::EnsembleKalmanProcess,
    g::M,
    Δt_new::NFT,
) where {M <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}}
    # when using g to calculate Δt, pass only successful particles through
    successful_ens, _ = split_indices_by_success(g)
    terminate = calculate_timestep!(ekp, g[:, successful_ens], Δt_new, get_scheduler(ekp))
    return terminate
end

function calculate_timestep!(
    ekp::EnsembleKalmanProcess,
    g::M,
    Δt_new::NFT,
    scheduler::DefaultScheduler,
) where {M <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}}
    if !isnothing(Δt_new)
        push!(get_Δt(ekp), Δt_new)
    else
        push!(get_Δt(ekp), scheduler.Δt_default)
    end
    nothing
end

function calculate_timestep!(
    ekp::EnsembleKalmanProcess,
    g::M,
    Δt_new::NFT,
    scheduler::MutableScheduler,
) where {M <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}}
    if !isnothing(Δt_new)
        push!(get_Δt(ekp), Δt_new)
        push!(scheduler.Δt_mutable, Δt_new) # change final stored timestep value
    elseif isnothing(Δt_new) && isempty(get_Δt(ekp))
        push!(get_Δt(ekp), 1.0)
        push!(scheduler.Δt_mutable, 1.0) # change final stored timestep value
    else
        push!(get_Δt(ekp), scheduler.Δt_mutable[end])
    end
    nothing
end

function calculate_timestep!(
    ekp::EnsembleKalmanProcess{IT, FT, SS},
    g::MM,
    Δt_new::NFT,
    scheduler::EKSStableScheduler,
) where {MM <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}, IT, FT, SS <: Sampler}
    if !isnothing(Δt_new)
        @info "Cannot override EKSStableScheduler-type timestep selection, ignoring Δt_new = $(Δt_new)"
    end

    # g_mean: 1 x N_obs
    M, J = size(g)
    g_mean = mean(g, dims = 2)
    y_mean = get_obs(ekp)
    Γ_inv = get_obs_noise_cov_inv(ekp)
    D = (1 / J) * ((g .- g_mean)' * Γ_inv * (g .- y_mean))
    numerator = max(scheduler.numerator, eps())
    nugget = max(scheduler.nugget, eps())


    Δt = numerator / (norm(D) + nugget)
    push!(get_Δt(ekp), Δt)
    nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes square matrix `mat` positive definite, by symmetrizing and bounding the minimum eigenvalue below by `tol`
"""
function posdef_correct(mat::AbstractMatrix; tol::Real = 1e8 * eps())
    mat = deepcopy(mat)
    if !issymmetric(mat)
        out = 0.5 * (mat + permutedims(mat, (2, 1))) #symmetrize
        if isposdef(out)
            # very often, small numerical errors cause asymmetry, so cheaper to add this branch
            return out
        end
    else
        out = mat
    end

    if !isposdef(out)
        nugget = abs(minimum(eigvals(out)))
        for i in 1:size(out, 1)
            out[i, i] += nugget + tol # add to diag
        end
    end
    return out
end


# Iglesias Yan 2021 paper
function calculate_timestep!(
    ekp::EnsembleKalmanProcess,
    g::MM,
    Δt_new::NFT,
    scheduler::DataMisfitController,
) where {MM <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}}
    if !isnothing(Δt_new)
        @info "Cannot override DataMisfitController-type timestep selection, ignoring Δt_new = $(Δt_new)"
    end

    M, J = size(g)
    T = scheduler.terminate_at

    # check if no minibatching
    os = get_observation_series(ekp)
    index = get_current_minibatch_index(os)
    len_epoch = length(get_minibatches(os)[index["epoch"]])

    if isempty(get_Δt(ekp))
        push!(scheduler.iteration, 1)
        inv_sqrt_Γ = sqrt(posdef_correct(get_obs_noise_cov_inv(ekp)))
        push!(scheduler.inv_sqrt_noise, inv_sqrt_Γ)
    elseif len_epoch == 1 # only no minibatching
        scheduler.iteration[end] += 1
        inv_sqrt_Γ = scheduler.inv_sqrt_noise[end]
    else
        scheduler.iteration[end] += 1
        inv_sqrt_Γ = sqrt(posdef_correct(get_obs_noise_cov_inv(ekp)))
        scheduler.inv_sqrt_noise[1] = inv_sqrt_Γ
    end
    n = scheduler.iteration[end]
    sum_Δt = (n == 1) ? 0.0 : sum(get_Δt(ekp))
    sum_Δt_min1 = (n <= 2) ? 0.0 : sum(get_Δt(ekp)[1:(end - 1)])
    # On termination condition:
    if sum_Δt >= T
        if sum_Δt_min1 < T # "Just reached termination"
            if scheduler.on_terminate == "stop"
                @warn "Termination condition of scheduler `DataMisfitController` has been exceeded, returning `true` from `update_ensemble!` and preventing futher updates\n Set on_terminate=\"continue\" in `DataMisfitController` to ignore termination"
                return true #returns a terminate call
            elseif scheduler.on_terminate == "continue_fixed"
                @warn "Termination condition of scheduler `DataMisfitController` has been exceeded. \non_terminate=\"continue_fixed\" selected. Proceeding with the final fixed timestep of $(get_Δt(ekp)[end])."
            elseif scheduler.on_terminate == "continue"
                @warn "Termination condition of scheduler `DataMisfitController` has been exceeded. \non_terminate=\"continue\" selected. Proceeding with algorithm"
            end
        end
        if scheduler.on_terminate == "continue_fixed"
            push!(get_Δt(ekp), get_Δt(ekp)[end])
            return nothing
        end
    end

    y_mean = get_obs(ekp)

    Φ = [0.5 * norm(inv_sqrt_Γ * (g[:, j] - reshape(y_mean, :, 1)))^2 for j in 1:J]
    Φ_mean = mean(Φ)
    Φ_var = var(Φ)

    q = maximum((M / (2 * Φ_mean), sqrt(M / (2 * Φ_var))))

    if sum_Δt < T
        Δt = minimum([q, T - sum_Δt])
    else # when termination condition satisfied but choose to continue
        Δt = q
    end

    # in theory the following should be the same.
    push!(get_Δt(ekp), Δt)

    if (sum_Δt < T) && (sum_Δt + Δt >= T)
        @info "Termination condition of scheduler `DataMisfitController` will be exceeded during the next iteration."
    end
    nothing
end

# overload ==
function Base.:(==)(lrs_a::LRS, lrs_b::LRS) where {LRS <: LearningRateScheduler}
    checks = [false for i in 1:length(fieldnames(LRS))]
    for (i, f) in enumerate(fieldnames(LRS))
        checks[i] = getfield(lrs_a, f) == getfield(lrs_b, f)
    end
    return all(checks)
end


##########################################################

function calculate_timestep!(
    ekp::EnsembleKalmanProcess{FT, IT, NRS},
    g::MM,
    Δt_new::NFT,
    scheduler::EKSStableScheduler,
    ) where {MM <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}, IT, FT, NRS <: NonreversibleSampler}
    if !isa(get_process(ekp), NonreversibleSampler)
        throw(ArgumentError("EKSNR sampler only constructable for `process=NonreversibleSampler()`"))
    end
    # compute preconditioners! 
    tol = 1e8 * eps()
    # u_mean: N_par × 1
    # g_mean: N_obs × 1
    u = get_u_final(ekp)
    u_mean = mean(u, dims = 2)
    dimen = size(u_mean, 1)
    g_mean = mean(g, dims = 2)
    cov_uu, cov_ug, cov_gg = get_cov_blocks(cov([u; g], dims = 2, corrected = false), dimen)

    # Build the preconditioners:
    decomp_Cuu = eigen(cov_uu)
    # .values = min -> max evals,
    # .vectors stores evecs as rows
    # D̃_opt = d * λ_min⁻¹ * v_min ⊗ v_min = λ_min⁻¹ D_opt
    λ_min_un = decomp_Cuu.values[1]
    v = decomp_Cuu.vectors[:, 1]
    
    # normalize
    v /= norm(v) # normalize v
    λ_min = norm(v) * λ_min_un #rescale lambda for a normlized v
    λ_opt = 1 / λ_min


    D_opt = Symmetric(dimen * v * v')
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
            Ĵ_opt[j, k] = (λ[j] + λ[k]) / (λ[j] - λ[k]) * (1 / λ_min)
            Ĵ_opt[k, j] = -Ĵ_opt[j, k]
        end
    end
    Ψ=Ψ'
    sqC = sqrt(cov_uu)
    # as ΨΨ' = I
    J_opt = sqC * Ψ * Ĵ_opt * Ψ' * sqC
    # g_mean: 1 x N_obs
    M, J = size(g)
    y_mean = get_obs(ekp)
    Γ_inv = get_obs_noise_cov_inv(ekp)
    
    
    #DD = (D_opt + J_opt) * (cov_uu \ cov_ug * Γ_inv * (g .- y_mean))
    # could we use that (D_opt+J_opt)C^{-1} < ubd*I
    # Then setting dt = 1/||ubd...|| < 1/||DD|| is conservative
    # paper bound of (D+J)*inv(C)
    # lambda_opt*(d+sqrt(cond(C))*(2*pi*c^2)/(sqrt(3)(c^2-1))*sqrt(d)(d-1)
    ubd_paper = λ_opt * (dimen + sqrt(cond(cov_uu)) * 2*pi*prefactor^2/(sqrt(3)*(prefactor^2-1))*sqrt(dimen)*(dimen-1))
    DpJCinv = (D_opt + J_opt)*inv(cov_uu)
    @info "$(ubd) < $(ubd_paper)"

    ubd = ubd_paper
    DD = ubd * (1 / J) * ((g .- g_mean)' * Γ_inv * (g .- y_mean))   

    numerator = max(scheduler.numerator, eps())
    nugget = max(scheduler.nugget, eps())
    # Δt = numerator * norm(u_mean)  / (norm(DD) + nugget)

    Δt = numerator / (norm(DD) + nugget)

    push!(get_Δt(ekp), Δt)
    nothing
end

