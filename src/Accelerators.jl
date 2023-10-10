# included in EnsembleKalmanProcess.jl

export DefaultAccelerator, NesterovAccelerator
export update_state!, set_initial_acceleration!

"""
$(TYPEDEF)

Default accelerator provides no acceleration, runs traditional EKI
"""
struct DefaultAccelerator <: Accelerator end

"""
$(TYPEDEF)

Accelerator that adapts Nesterov's momentum method for EKI. 
Stores a previous state value u_prev for computational purposes (note this is distinct from state returned as "ensemble value")

$(TYPEDFIELDS)
"""
mutable struct NesterovAccelerator{FT <: AbstractFloat} <: Accelerator
    r::FT
    u_prev::Any
end

function NesterovAccelerator(r = 3.0, initial = Float64[])
    return NesterovAccelerator(r, initial)
end


"""
Sets u_prev to the initial parameter values
"""
function set_ICs!(accelerator::NesterovAccelerator{FT}, u::MA) where {FT <: AbstractFloat, MA <: AbstractMatrix{FT}}
    accelerator.u_prev = u
end


"""
Performs traditional state update with no momentum. 
"""
function update_state!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, DefaultAccelerator},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix{FT}}
    push!(ekp.u, DataContainer(u, data_are_columns = true))
end

"""
Performs state update with modified Nesterov momentum approach.
"""
function update_state!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Process, LRS <: LearningRateScheduler, MA <: AbstractMatrix{FT}}
    ## update "v" state:
    k = get_N_iterations(ekp) + 2
    v = u .+ (1 - ekp.accelerator.r / k) * (u .- ekp.accelerator.u_prev)

    ## update "u" state: 
    ekp.accelerator.u_prev = u

    ## push "v" state to EKP object
    push!(ekp.u, DataContainer(v, data_are_columns = true))
end


"""
State update method for UKI with no acceleration.
The Accelerator framework has not yet been integrated with UKI process;
UKI tracks its own states, so this method is empty.
"""
function update_state!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, DefaultAccelerator},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, MA <: AbstractMatrix{FT}}

end

"""
Placeholder state update method for UKI with Nesterov Accelerator.
The Accelerator framework has not yet been integrated with UKI process, so this
method throws an error.
"""
function update_state!(
    ekp::EnsembleKalmanProcess{FT, IT, P, LRS, NesterovAccelerator{FT}},
    u::MA,
) where {FT <: AbstractFloat, IT <: Int, P <: Unscented, LRS <: LearningRateScheduler, MA <: AbstractMatrix{FT}}
    throw(
        ArgumentError(
            "option `accelerator = NesterovAccelerator` is not implemented for UKI, please use `DefaultAccelerator`",
        ),
    )
end
