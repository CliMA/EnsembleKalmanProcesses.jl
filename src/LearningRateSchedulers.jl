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
        push!(ekp.Δt, Δt_new)
    else
        push!(ekp.Δt, scheduler.Δt_default)
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
        push!(ekp.Δt, Δt_new)
        push!(scheduler.Δt_mutable, Δt_new) # change final stored timestep value
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, 1.0)
        push!(scheduler.Δt_mutable, 1.0) # change final stored timestep value
    else
        push!(ekp.Δt, scheduler.Δt_mutable[end])
    end
    nothing
end

function calculate_timestep!(
    ekp::EnsembleKalmanProcess,
    g::MM,
    Δt_new::NFT,
    scheduler::EKSStableScheduler,
) where {MM <: AbstractMatrix, NFT <: Union{Nothing, AbstractFloat}}
    if !isnothing(Δt_new)
        @info "Cannot override EKSStableScheduler-type timestep selection, ignoring Δt_new = $(Δt_new)"
    end

    # g_mean: 1 x N_obs
    M, J = size(g)
    g_mean = mean(g, dims = 2)
    y_mean = ekp.obs_mean
    if isa(ekp.obs_noise_cov, UniformScaling)
        Γ = ekp.obs_noise_cov.λ * I(M) # converts into MxM matrix, 
    else
        Γ = ekp.obs_noise_cov
    end
    # (G(u) - E[G(u)])ᵀΓ⁻¹(G(u) - E[y])
    D = (1 / J) * ((g .- g_mean)' * (Γ \ (g .- y_mean)))

    numerator = max(scheduler.numerator, eps())
    nugget = max(scheduler.nugget, eps())


    Δt = numerator / (norm(D) + nugget)
    push!(ekp.Δt, Δt)
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

    nugget = abs(minimum(eigvals(out)))
    for i in 1:size(out, 1)
        out[i, i] += nugget + tol #add to diag
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

    if isempty(ekp.Δt)
        push!(scheduler.iteration, 1)
        if isa(ekp.obs_noise_cov, UniformScaling)
            Γ = ekp.obs_noise_cov.λ * I(M) # converts into MxM matrix
        else
            Γ = ekp.obs_noise_cov
        end
        inv_sqrt_Γ = inv(sqrt(posdef_correct(Γ)))
        push!(scheduler.inv_sqrt_noise, inv_sqrt_Γ)
    else
        scheduler.iteration[end] += 1
        inv_sqrt_Γ = scheduler.inv_sqrt_noise[end]
    end
    n = scheduler.iteration[end]
    sum_Δt = (n == 1) ? 0.0 : sum(ekp.Δt)
    sum_Δt_min1 = (n <= 2) ? 0.0 : sum(ekp.Δt[1:(end - 1)])
    # On termination condition:
    if sum_Δt >= T
        if sum_Δt_min1 < T # "Just reached termination"
            if scheduler.on_terminate == "stop"
                @warn "Termination condition of timestepping scheme `DataMisfitController` has been exceeded. Preventing futher updates\n Set on_terminate=\"continue\" in `DataMisfitController` to ignore termination"
                return true #returns a terminate call
            elseif scheduler.on_terminate == "continue_fixed"
                @warn "Termination condition of timestepping scheme `DataMisfitController` has been exceeded. \non_terminate=\"continue_fixed\" selected. Proceeding with the final fixed timestep of $(ekp.Δt[end])."
            elseif scheduler.on_terminate == "continue"
                @warn "Termination condition of timestepping scheme `DataMisfitController` has been exceeded. \non_terminate=\"continue\" selected. Proceeding with algorithm"
            end
        end
        if scheduler.on_terminate == "continue_fixed"
            push!(ekp.Δt, ekp.Δt[end])
            return nothing
        end
    end

    y_mean = ekp.obs_mean

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
    push!(ekp.Δt, Δt)

    if (sum_Δt < T) && (sum_Δt + Δt >= T)
        @info "Termination condition of timestepping scheme `DataMisfitController` will be exceeded during the next iteration."
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
