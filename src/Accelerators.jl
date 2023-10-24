# included in EnsembleKalmanProcess.jl

export DefaultAccelerator, NesterovAccelerator, ConstantStepNesterovAccelerator
export accelerate!, set_initial_acceleration!

"""
$(TYPEDEF)

Default accelerator provides no acceleration, runs traditional EKI
"""
struct DefaultAccelerator <: Accelerator end



"""
$(TYPEDEF)

Accelerator that adapts a Constant-timestep version of Nesterov's momentum method for EKI. 
Stores a previous state value u_prev for computational purposes (note this is distinct from state returned as "ensemble value")

$(TYPEDFIELDS)
"""
mutable struct ConstantStepNesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    r::FT
    u_prev::Any
end

function ConstantStepNesterovAccelerator(r = 3.0, initial = Float64[])
    return ConstantStepNesterovAccelerator(r, initial)
end


"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(
    accelerator::ConstantStepNesterovAccelerator{FT},
    u::MA,
) where {FT <: AbstractFloat, MA <: AbstractMatrix}
    accelerator.u_prev = u
end


"""
$(TYPEDEF)

Accelerator that adapts Nesterov's momentum method for EKI. 
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
Performs state update with modified Nesterov momentum approach.
The dependence of the momentum parameter for variable timestep can be found e.g. here "https://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf"
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    ## update "v" state:
    #v = u .+ 2 / (get_N_iterations(ekp) + 2) * (u .- ekp.accelerator.u_prev)
    Δt_prev = length(ekp.Δt) == 1 ? 1 : ekp.Δt[end - 1]
    Δt = ekp.Δt[end]
    θ_prev = ekp.accelerator.θ_prev

    # condition θ_prev^2 * (1 - θ) * Δt \leq Δt_prev * θ^2
    a = sqrt(θ_prev^2 * Δt / Δt_prev)
    θ = (-a + sqrt(a^2 + 4)) / 2

    v = u .+ θ * (1 / θ_prev - 1) * (u .- ekp.accelerator.u_prev)

    ## update "u" state: 
    ekp.accelerator.u_prev = u
    ekp.accelerator.θ_prev = θ

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end


"""
State update method for UKI with Nesterov Accelerator.
The dependence of the momentum parameter for variable timestep can be found e.g. here "https://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf"
Performs identical update as with other methods, but requires reconstruction of mean and covariance of the accelerated positions prior to saving.
"""
function accelerate!(
    uki::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::AM,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, AM <: AbstractMatrix}

    #identical update stage as before
    ## update "v" state:
    Δt_prev = length(uki.Δt) == 1 ? 1 : uki.Δt[end - 1]
    Δt = uki.Δt[end]
    θ_prev = uki.accelerator.θ_prev


    # condition θ_prev^2 * (1 - θ) * Δt \leq Δt_prev * θ^2
    a = sqrt(θ_prev^2 * Δt / Δt_prev)
    θ = (-a + sqrt(a^2 + 4)) / 2

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
Performs state update with modified constant timestep Nesterov momentum approach.
"""
function accelerate!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, ConstantStepNesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix}
    ## update "v" state:
    k = get_N_iterations(ekp) + 2
    v = u .+ (1 - ekp.accelerator.r / k) * (u .- ekp.accelerator.u_prev)

    ## update "u" state: 
    ekp.accelerator.u_prev = u

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end


"""
State update method for UKI with constant timestep Nesterov Accelerator.
Performs identical update as with other methods, but requires reconstruction of mean and covariance of the accelerated positions prior to saving.
"""
function accelerate!(
    uki::EnsembleKalmanProcess{FT, IT, P, LRS, ConstantStepNesterovAccelerator{FT}},
    u::AM,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, AM <: AbstractMatrix}

    #identical update stage as before
    ## update "v" state:
    k = get_N_iterations(uki) + 2
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
