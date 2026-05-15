# included in EnsembleKalmanProcess.jl

export DefaultAccelerator, ConstantNesterovAccelerator, NesterovAccelerator, FirstOrderNesterovAccelerator
export accelerate!

"""
$(TYPEDEF)

Default accelerator provides no acceleration, runs traditional EKI
"""
struct DefaultAccelerator <: Accelerator end


"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter is a constant value ``λ``, the default of 0.9.

$(TYPEDFIELDS)
"""
mutable struct ConstantNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    "constant momentum parameter λ ∈ (0, 1)"
    λ::FT
    "ensemble parameter matrix from the previous iteration, used to compute the momentum update"
    u_prev::Array{FT}
end

"""
$(TYPEDSIGNATURES)

Construct a `ConstantNesterovAccelerator` with momentum parameter `λ` and optional initial ensemble state `initial`.
"""
function ConstantNesterovAccelerator(λ = 0.9, initial = Float64[])
    return ConstantNesterovAccelerator(λ, initial)
end

"""
$(TYPEDSIGNATURES)

Set the previous-state cache `u_prev` to the initial ensemble matrix `u`.
"""
function set_ICs!(accelerator::ConstantNesterovAccelerator{FT}, u::MA) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end


"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter at iteration ``k`` is given by ``1-\\frac{r}{k+2}``, a first-order representation of the asymptotic schedule ``1-\\frac{r}{k}-\\mathcal{O}(k^{-2})`` advocated in [Su, Boyd, Candes (2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/f09696910bdd874a99cd74c8f05b5c44-Paper.pdf). Stores a previous ensemble state for momentum computation; this is distinct from the state returned as the ensemble value.

$(TYPEDFIELDS)
"""
mutable struct FirstOrderNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    "decay exponent r controlling the momentum schedule; the momentum coefficient at iteration k is 1 - r/(k+2)"
    r::FT
    "ensemble parameter matrix from the previous iteration, used to compute the momentum update"
    u_prev::Array{FT}
end

"""
$(TYPEDSIGNATURES)

Construct a `FirstOrderNesterovAccelerator` with decay exponent `r` and optional initial ensemble state `initial`.
"""
function FirstOrderNesterovAccelerator(r = 3.0, initial = Float64[])
    return FirstOrderNesterovAccelerator(r, initial)
end


"""
$(TYPEDSIGNATURES)

Set the previous-state cache `u_prev` to the initial ensemble matrix `u`.
"""
function set_ICs!(
    accelerator::FirstOrderNesterovAccelerator{FT},
    u::MA,
) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end


"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI using the recurrence ``\\theta_{k+1} = \\frac{-\\theta_k^2 + \\sqrt{\\theta_k^4 + 4\\theta_k^2}}{2}`` with ``\\theta_0 = 1``, giving momentum coefficient ``\\theta_{k+1}(\\theta_k^{-1} - 1)`` at each iteration. This recurrence is also described in [Su, Boyd, Candes (2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/f09696910bdd874a99cd74c8f05b5c44-Paper.pdf) and is the recommended variant. Stores a previous ensemble state for momentum computation; this is distinct from the state returned as the ensemble value.

$(TYPEDFIELDS)
"""
mutable struct NesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    "ensemble parameter matrix from the previous iteration, used to compute the momentum update"
    u_prev::Array{FT}
    "previous value of the Nesterov momentum parameter θ"
    θ_prev::FT
end

"""
$(TYPEDSIGNATURES)

Construct a `NesterovAccelerator` with optional initial ensemble state `initial` and initial momentum parameter ``\\theta_0 = 1``.
"""
function NesterovAccelerator(initial = Float64[])
    return NesterovAccelerator(initial, 1.0)
end


"""
$(TYPEDSIGNATURES)

Set the previous-state cache `u_prev` to the initial ensemble matrix `u`.
"""
function set_ICs!(accelerator::NesterovAccelerator{FT}, u::MA) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end


"""
$(TYPEDSIGNATURES)

Push ensemble state `u` unchanged into `ekp`, performing no momentum acceleration.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, DefaultAccelerator},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    push!(ekp.u, DataContainer(u, data_are_columns = true))
end

"""
$(TYPEDSIGNATURES)

Apply the `NesterovAccelerator` momentum update to ensemble state `u` and push the accelerated state into `ekp`.

The momentum coefficient follows the recurrence described in [`NesterovAccelerator`](@ref); see also the [lecture notes](https://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf) for background.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    ## update "v" state:
    Δt_prev = length(get_Δt(ekp)) == 1 ? get_Δt(ekp)[end] : get_Δt(ekp)[end - 1]
    Δt = get_Δt(ekp)[end]
    θ_prev = get_accelerator(ekp).θ_prev

    b = θ_prev^2
    θ = (-b + sqrt(b^2 + 4 * b)) / 2

    v = u .+ θ * (1 / θ_prev - 1) * (u .- get_accelerator(ekp).u_prev)
    ## update "u" state: 
    get_accelerator(ekp).u_prev = u
    get_accelerator(ekp).θ_prev = θ

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end


"""
$(TYPEDSIGNATURES)

Apply the `NesterovAccelerator` momentum update to UKI ensemble state `u` and push the accelerated state into `uki`.

Performs the same momentum step as the generic method, then reconstructs the ensemble mean and covariance of the accelerated sigma points by inverting the UKI prediction transformation, and overwrites the stored `u_mean` and `uu_cov`.
"""
function accelerate!(
    uki::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::AM,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, AM <: AbstractMatrix}

    #identical update stage as before
    ## update "v" state:
    Δt_prev = length(uki.Δt) == 1 ? uki.Δt[end] : uki.Δt[end - 1]
    Δt = uki.Δt[end]
    θ_prev = uki.accelerator.θ_prev

    b = θ_prev^2
    θ = (-b + sqrt(b^2 + 4 * b)) / 2

    v = u .+ θ * (1 / θ_prev - 1) * (u .- uki.accelerator.u_prev)

    ## update "u" state: 
    uki.accelerator.u_prev = u
    uki.accelerator.θ_prev = θ

    ## push "v" state to UKI object
    push!(uki.u, DataContainer(v, data_are_columns = true))

    # additional complication: the stored "u_mean" and "uu_cov" are not the mean/cov of this ensemble
    # the ensemble comes from the prediction operation acted upon this. we invert the prediction of the mean/cov of the sigma ensemble from u_mean/uu_cov
    # u_mean = 1/alpha*(mean(v) - r) + r
    # uu_cov = 1/alpha^2*(cov(v) - Σ_ω)
    α_reg = uki.process.α_reg
    r = uki.process.r
    Σ_ω = uki.process.Σ_ω
    Δt = uki.Δt[end]

    v_mean = construct_mean(uki, v)
    vv_cov = construct_cov(uki, v, v_mean)
    u_mean = 1 / α_reg * (v_mean - r) + r
    uu_cov = (1 / α_reg)^2 * (vv_cov - Σ_ω * Δt)

    # overwrite the saved u_mean/uu_cov
    uki.process.u_mean[end] = u_mean # N_ens x N_params 
    uki.process.uu_cov[end] = uu_cov # N_ens x N_data

end



"""
$(TYPEDSIGNATURES)

Apply the `ConstantNesterovAccelerator` momentum update to ensemble state `u` and push the accelerated state into `ekp`.

The momentum coefficient is the constant λ stored in the accelerator.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, ConstantNesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    ## update "v" state:
    v = u .+ get_accelerator(ekp).λ * (u .- get_accelerator(ekp).u_prev)

    ## update "u" state: 
    get_accelerator(ekp).u_prev = u

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end

"""
$(TYPEDSIGNATURES)

Apply the `ConstantNesterovAccelerator` momentum update to UKI ensemble state `u` and push the accelerated state into `uki`.

Performs the same constant-momentum step as the generic method, then reconstructs the ensemble mean and covariance of the accelerated sigma points by inverting the UKI prediction transformation, and overwrites the stored `u_mean` and `uu_cov`.
"""
function accelerate!(
    uki::EnsembleKalmanProcess{FT, IT, P, LRS, ConstantNesterovAccelerator{FT}},
    u::AM,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, AM <: AbstractMatrix}

    #identical update stage as before
    ## update "v" state:
    v = u .+ uki.accelerator.λ * (u .- uki.accelerator.u_prev)

    ## update "u" state: 
    uki.accelerator.u_prev = u

    ## push "v" state to UKI object
    push!(uki.u, DataContainer(v, data_are_columns = true))

    # additional complication: the stored "u_mean" and "uu_cov" are not the mean/cov of this ensemble
    # the ensemble comes from the prediction operation acted upon this. we invert the prediction of the mean/cov of the sigma ensemble from u_mean/uu_cov
    # u_mean = 1/alpha*(mean(v) - r) + r
    # uu_cov = 1/alpha^2*(cov(v) - Σ_ω)
    α_reg = uki.process.α_reg
    r = uki.process.r
    Σ_ω = uki.process.Σ_ω
    Δt = uki.Δt[end]

    v_mean = construct_mean(uki, v)
    vv_cov = construct_cov(uki, v, v_mean)
    u_mean = 1 / α_reg * (v_mean - r) + r
    uu_cov = (1 / α_reg)^2 * (vv_cov - Σ_ω * Δt)

    # overwrite the saved u_mean/uu_cov
    uki.process.u_mean[end] = u_mean # N_ens x N_params 
    uki.process.uu_cov[end] = uu_cov # N_ens x N_data

end

"""
$(TYPEDSIGNATURES)

Apply the `FirstOrderNesterovAccelerator` momentum update to ensemble state `u` and push the accelerated state into `ekp`.

The momentum coefficient at iteration k is ``1 - r/(k+2)``, where r is the decay exponent stored in the accelerator.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, FirstOrderNesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    ## update "v" state:
    k = get_N_iterations(ekp) + 3 # get_N_iterations starts at 0
    v = u .+ (1 - get_accelerator(ekp).r / k) * (u .- get_accelerator(ekp).u_prev)

    ## update "u" state: 
    get_accelerator(ekp).u_prev = u

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end


"""
$(TYPEDSIGNATURES)

Apply the `FirstOrderNesterovAccelerator` momentum update to UKI ensemble state `u` and push the accelerated state into `uki`.

Performs the same iteration-dependent momentum step as the generic method, then reconstructs the ensemble mean and covariance of the accelerated sigma points by inverting the UKI prediction transformation, and overwrites the stored `u_mean` and `uu_cov`.
"""
function accelerate!(
    uki::EnsembleKalmanProcess{FT, IT, P, LRS, FirstOrderNesterovAccelerator{FT}},
    u::AM,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, AM <: AbstractMatrix}

    #identical update stage as before
    ## update "v" state:
    k = get_N_iterations(uki) + 3  # get_N_iterations starts at 0
    v = u .+ (1 - uki.accelerator.r / k) * (u .- uki.accelerator.u_prev)

    ## update "u" state: 
    uki.accelerator.u_prev = u

    ## push "v" state to UKI object
    push!(uki.u, DataContainer(v, data_are_columns = true))

    # additional complication: the stored "u_mean" and "uu_cov" are not the mean/cov of this ensemble
    # the ensemble comes from the prediction operation acted upon this. we invert the prediction of the mean/cov of the sigma ensemble from u_mean/uu_cov
    # u_mean = 1/alpha*(mean(v) - r) + r
    # uu_cov = 1/alpha^2*(cov(v) - Σ_ω)
    α_reg = uki.process.α_reg
    r = uki.process.r
    Σ_ω = uki.process.Σ_ω
    Δt = uki.Δt[end]

    v_mean = construct_mean(uki, v)
    vv_cov = construct_cov(uki, v, v_mean)
    u_mean = 1 / α_reg * (v_mean - r) + r
    uu_cov = (1 / α_reg)^2 * (vv_cov - Σ_ω * Δt)

    # overwrite the saved u_mean/uu_cov
    uki.process.u_mean[end] = u_mean # N_ens x N_params 
    uki.process.uu_cov[end] = uu_cov # N_ens x N_data

end
