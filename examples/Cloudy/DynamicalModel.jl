module DynamicalModel

using DocStringExtensions

using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.Sources

using Random
using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations

export ModelSettings
export run_dyn_model

"""
    ModelSettings{FT<:AbstractFloat, KT, D}

Structure to hold all information to run the dynamical model

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ModelSettings{FT <: AbstractFloat, KT, D}
    "a kernel tensor specifying the physics of collision-coalescence"
    kernel::KT
    "a cloud droplet mass distribution function"
    dist::D
    "the moments of `dist` that the model should return"
    moments::Array{FT, 1}
    "time period over which to run the model, e.g., `(0, 1)`"
    tspan::Tuple{FT, FT}
end


"""
    run_dyn_model(ϕ::Array{FT, 1}, settings::ModelSettings{FT}) where {FT<:AbstractFloat}

Run the dynamical model (Cloudy) for the given parameter vector ϕ. 
Return a vector of length N_obs containing the model output

 - `ϕ` - parameter vector of length N_parameters
 - `settings` - a ModelSettings struct

"""
function run_dyn_model(ϕ::Array{FT, 1}, settings::ModelSettings{FT}) where {FT <: AbstractFloat}

    # generate the initial distribution
    dist = ParticleDistributions.update_params(settings.dist, ϕ)

    # Numerical parameters
    tol = FT(1e-7)

    # Make sure moments are up to date. mom0 is the initial condition for the
    # ODE problem
    moments = settings.moments
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = ParticleDistributions.moment(dist, convert(FT, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, par, t) = get_int_coalescence(M, par, settings.kernel)
    ODE_par = Dict(:dist => dist)  # ODE parameters
    prob = ODEProblem(rhs, moments_init, settings.tspan, ODE_par)
    # Solve the ODE
    sol = solve(prob, CVODE_BDF(), alg_hints = [:stiff], reltol = tol, abstol = tol)
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]

    return moments_final
end

end # module
