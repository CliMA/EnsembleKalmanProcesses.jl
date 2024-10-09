# included in EnsembleKalmanProcess.jl

export DefaultAccelerator, ConstantNesterovAccelerator, NesterovAccelerator, FirstOrderNesterovAccelerator, AffineInvariantNesterovAccelerator
export accelerate!, set_initial_acceleration!

"""
$(TYPEDEF)

Default accelerator provides no acceleration, runs traditional EKI
"""
struct DefaultAccelerator <: Accelerator end


"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter is a constant value ``λ``, the default of 0.9. 
"""
mutable struct ConstantNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    λ::FT
    u_prev::Array{FT}
end

function ConstantNesterovAccelerator(λ = 0.9, initial = Float64[])
    return ConstantNesterovAccelerator(λ, initial)
end

"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(accelerator::ConstantNesterovAccelerator{FT}, u::MA) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end

"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter at iteration ``k``, is given by ``1-\\frac{3}{k+2}``. This variant is a first order repesentation of the desired asymptotic behavior ``1-\\frac{3}{k}- \\mathcal{O}(\\frac{1}{k^2})`` advocated in [(Su, Boyd, Candes, 2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/f09696910bdd874a99cd74c8f05b5c44-Paper.pdf).
Stores a previous state value u_prev for computational purposes (note this is distinct from state returned as "ensemble value")

$(TYPEDFIELDS)
"""
mutable struct FirstOrderNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    r::FT
    u_prev::Array{FT}
end

function FirstOrderNesterovAccelerator(r = 3.0, initial = Float64[])
    return FirstOrderNesterovAccelerator(r, initial)
end


"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(
    accelerator::FirstOrderNesterovAccelerator{FT},
    u::MA,
) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end



"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter is given by ``\\theta_{k+1}(\\frac{1}{\\theta_{k}} - 1)`` where ``\\theta_{k+1} = \\frac{-\\theta_k^2 + \\sqrt{\\theta_k^4 + 4 \\theta_k^2}}{2}`` and ``\\theta_0 = 1``. This recurrence, also mentioned in [(Su, Boyd, Candes, 2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/f09696910bdd874a99cd74c8f05b5c44-Paper.pdf), is our recommended variant.
Stores a previous state value u_prev for computational purposes (note this is distinct from state returned as "ensemble value")

$(TYPEDFIELDS)
"""
mutable struct NesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    u_prev::Array{FT}
    θ_prev::FT
end

function NesterovAccelerator(initial = Float64[])
    return NesterovAccelerator(initial, 1.0)
end


"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(accelerator::NesterovAccelerator{FT}, u::MA) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end


"""
Performs traditional state update with no momentum. 
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, DefaultAccelerator},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    push!(ekp.u, DataContainer(u, data_are_columns = true))
end

"""
Performs state update with Nesterov momentum approach.
The dependence of the momentum parameter can be found e.g. here "https://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf"
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
State update method for UKI with Nesterov Accelerator.
The dependence of the momentum parameter can be found e.g. here "https://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf"
Performs identical update as with other methods, but requires reconstruction of mean and covariance of the accelerated positions prior to saving.
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
Performs state update with modified constant Nesterov Accelerator.
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
State update method for UKI with constant Nesterov Accelerator.
Performs identical update as with other methods, but requires reconstruction of mean and covariance of the accelerated positions prior to saving.
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
Performs state update with modified first-order Nesterov Accelerator.
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
State update method for UKI with first-order Nesterov Accelerator.
Performs identical update as with other methods, but requires reconstruction of mean and covariance of the accelerated positions prior to saving.
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


####################################################
####################################################


"""
$(TYPEDEF)

Accelerator that adapts a variant of Nesterov's momentum method for EKI. In this variant the momentum parameter at iteration ``k``, is given by ``1-\\frac{3}{k+2}``. This variant also uses a nudge based on the underlying Geometry of the Covariance matrices

$(TYPEDFIELDS)
"""
mutable struct AffineInvariantNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    r::FT
    u_prev::Array{FT}
end

function AffineInvariantNesterovAccelerator(r = 3.0, initial = Float64[])
    return AffineInvariantNesterovAccelerator(r, initial)
end


"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(
    accelerator::AffineInvariantNesterovAccelerator{FT},
    u::MA,
) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end

"""
Performs state update with modified first-order Nesterov Accelerator.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, AffineInvariantNesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    
    ## update "v" state:
    k = get_N_iterations(ekp) + 3 # get_N_iterations starts at 0
    β = (1 - get_accelerator(ekp).r / k)

    # get the distribution of u and u_prev
    u_mean = mean(u,dims=2)
    u_cov = cov(u,dims=2)

    u_prev_mean = mean(ekp.accelerator.u_prev,dims=2)
    u_prev_cov = cov(ekp.accelerator.u_prev,dims=2)
    u_prev_sqcov = sqrt(u_prev_cov)
    u_prev_invsqcov = inv(u_prev_sqcov)

    # v = uₙ + β(uₙ-uₙ₋₁)
    v_mean = (1+β) * u_mean - β * u_prev_mean #lin interp
    # geodesic extension of covariance matrix from A->B by β using matrix sqrt
    # geod(A,B;1+β) = √A · ((√A)⁻¹ · B · (√A)⁻¹)¹⁺ᵝ · √A
    v_cov_geod = u_prev_sqcov * (u_prev_invsqcov * u_cov * u_prev_invsqcov)^(1+β) * u_prev_sqcov
    if !isposdef(v_cov)
        v_cov_geod = 0.5*(v_cov_geod+v_cov_geod')+1e-12*I
        @warn "v_cov not positive definite"
    end
    
    @info "U info\n ******** \n $β \n $u_cov \n $u_prev_cov"
    #    v = rand(MvNormal(v_mean[:], v_cov_geod), size(u,2))
    @info "Geodesic\n ******** \n $v_cov_geod"
    
    ###
    v_cov = (1+β)^2 * u_cov + β^2 * u_prev_cov
    @info "Indep-Gauss ***********\n $v_cov"
    v = u .+ (1 - get_accelerator(ekp).r / k) * (u .- get_accelerator(ekp).u_prev)
    @info "Particle\n ******** \n $(cov(v,dims=2))"

    ## update "u" state: 
    get_accelerator(ekp).u_prev = u

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end
