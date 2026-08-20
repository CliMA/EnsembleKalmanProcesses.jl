# Atmosphere-like multiscale Lorenz-96 toy model:
#   x_k      : synoptic/chaotic slow variables      (K of them)
#   y_{j,k}  : fast unresolved variables             (J per slow variable k)
#   (q,p)    : weakly coupled damped oscillator (intraseasonal variability)
#   c        : slow bistable "climate" variable (tipping element)
#
# State layout (flat vector u, length K + K*J + 3):
#   u[1:K]                  -> x
#   u[K+1 : K+K*J]          -> y, reshaped (J, K) so y[:, k] are the fast
#                              variables belonging to slow variable k
#   u[end-2], u[end-1], u[end] -> q, p, c
using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK # provides DP5(), the adaptive Dormand-Prince RK4(5) ("ode45") method

# Parameters, with names/defaults matching the "Initial parameter regime" table.
# μ is the (possibly time-dependent) exploratory forcing trajectory driving the
# bistable climate variable; it is stored as a callable so richer trajectories
# (not just the linear μ(t) = μ0 + r*t) can be substituted without touching the RHS.
struct L96MultiscaleParams{MF <: Function}
    K::Int
    J::Int
    F0::Float64
    χ::Float64
    hx::Float64
    hy::Float64
    ε::Float64
    ω::Float64
    γ::Float64
    α::Float64
    βq::Float64
    βp::Float64
    δ::Float64
    a::Float64
    λ::Float64
    ρ::Float64
    μ::MF
end

# Linear exploratory trajectory μ(t) = μ0 + r*t (the simplest case discussed in the text).
linear_mu(μ0::Real, r::Real) = t -> μ0 + r * t

function L96MultiscaleParams(;
    K::Int = 8,
    J::Int = 32,
    F0::Real = 18.0,
    χ::Real = 0.1,
    hx::Real = 1.0,
    hy::Real = 1.0,
    ε::Real = 0.1,
    ω::Real = 0.6,
    γ::Real = 0.02,
    α::Real = 0.1,
    βq::Real = 0.05,
    βp::Real = 0.05,
    δ::Real = 0.005,
    a::Real = 1.0,
    λ::Real = 0.05,
    ρ::Real = 0.05,
    μ::Function = linear_mu(0.0, 0.0),
)
    return L96MultiscaleParams(
        K,
        J,
        Float64(F0),
        Float64(χ),
        Float64(hx),
        Float64(hy),
        Float64(ε),
        Float64(ω),
        Float64(γ),
        Float64(α),
        Float64(βq),
        Float64(βp),
        Float64(δ),
        Float64(a),
        Float64(λ),
        Float64(ρ),
        μ,
    )
end

state_dim(p::L96MultiscaleParams) = p.K + p.K * p.J + 3

# Named views into a flat state vector (no copying).
@inline function unpack_state(u::AbstractVector, p::L96MultiscaleParams)
    K, J = p.K, p.J
    x = view(u, 1:K)
    y = reshape(view(u, (K + 1):(K + K * J)), J, K)
    q = u[K + K * J + 1]
    pp = u[K + K * J + 2]
    c = u[K + K * J + 3]
    return x, y, q, pp, c
end

# Convenience indices for pulling named components out of a saved trajectory matrix
# (state_dim x n_time), matching `unpack_state` above.
x_indices(p::L96MultiscaleParams) = 1:(p.K)
y_indices(p::L96MultiscaleParams) = (p.K + 1):(p.K + p.K * p.J)
q_index(p::L96MultiscaleParams) = p.K + p.K * p.J + 1
p_index(p::L96MultiscaleParams) = p.K + p.K * p.J + 2
c_index(p::L96MultiscaleParams) = p.K + p.K * p.J + 3

# In-place right-hand side, compatible with OrdinaryDiffEq.jl / DifferentialEquations.jl.
function l96_multiscale_rhs!(du::AbstractVector, u::AbstractVector, p::L96MultiscaleParams, t::Real)
    K, J = p.K, p.J
    x, y, q, pp, c = unpack_state(u, p)
    dx = view(du, 1:K)
    dy = reshape(view(du, (K + 1):(K + K * J)), J, K)

    xbar = mean(x)
    forcing_x = p.F0 + p.χ * c

    @inbounds for k in 1:K
        ybar_k = mean(view(y, :, k))
        km1 = mod1(k - 1, K)
        km2 = mod1(k - 2, K)
        kp1 = mod1(k + 1, K)
        dx[k] = (x[kp1] - x[km2]) * x[km1] - x[k] + forcing_x - p.hx * ybar_k
    end

    hy_mod = p.hy * (1 + p.α * q)
    @inbounds for k in 1:K
        xk = x[k]
        for j in 1:J
            jm1 = mod1(j - 1, J)
            jm2 = mod1(j - 2, J)
            jp1 = mod1(j + 1, J)
            dy[j, k] = ((y[jp1, k] - y[jm2, k]) * y[jm1, k] - y[j, k] + hy_mod * xk) / p.ε
        end
    end

    ybar = mean(y)
    du[K + K * J + 1] = p.ω * pp - p.γ * q + p.βq * xbar
    du[K + K * J + 2] = -p.ω * q - p.γ * pp + p.βp * xbar
    # δ multiplies (not divides) the RHS so that c is genuinely the slowest variable
    # (relaxation timescale ~ 1/δ = 200 MTU, matching Table 2), consistent with the fast
    # y-equation being sped up by dividing by the small parameter ε.
    du[K + K * J + 3] = p.δ * (p.μ(t) + p.a * c - c^3 + p.λ * xbar + p.ρ * ybar)
    return nothing
end

# Default initial condition: x near the forcing equilibrium with small noise, small-amplitude
# fast variables, oscillator at rest, and c on the positive stable branch of the (unforced)
# bistable well c^3 = a*c (roots 0, ±sqrt(a)).
function default_initial_condition(p::L96MultiscaleParams; rng::AbstractRNG = Random.default_rng(), x_scale::Real = 1.0, y_scale::Real = 0.1)
    u0 = zeros(state_dim(p))
    u0[x_indices(p)] .= p.F0 .+ x_scale .* randn(rng, p.K)
    u0[y_indices(p)] .= y_scale .* randn(rng, p.K * p.J)
    u0[q_index(p)] = 0.0
    u0[p_index(p)] = 0.0
    u0[c_index(p)] = sqrt(p.a)
    return u0
end

# Integrate the model with an adaptive embedded Runge-Kutta 4(5) scheme (Dormand-Prince,
# the classical "ode45"/"RK45" method), returning the OrdinaryDiffEq solution object.
function solve_l96_multiscale(
    p::L96MultiscaleParams,
    u0::AbstractVector,
    tspan::Tuple{<:Real, <:Real};
    solver = DP5(),
    reltol::Real = 1e-6,
    abstol::Real = 1e-6,
    saveat = nothing,
    maxiters::Integer = 10_000_000, # long, high-frequency (fast y / oscillator) runs need many adaptive steps
    kwargs...,
)
    prob = ODEProblem(l96_multiscale_rhs!, u0, tspan, p)
    if saveat === nothing
        return solve(prob, solver; reltol = reltol, abstol = abstol, maxiters = maxiters, kwargs...)
    else
        return solve(prob, solver; reltol = reltol, abstol = abstol, saveat = saveat, maxiters = maxiters, kwargs...)
    end
end
