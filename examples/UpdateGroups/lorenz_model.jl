struct LSettings
    # Number of longitude steps
    N::Int32
    # Timestep
    dt::Float64
    # end time
    t_end::Float64
    # Initial perturbation
    initial::Union{Array{Float64}, Nothing}

end

struct LParams{FT <: AbstractFloat, VV <: AbstractVector}
    # Slow - Mean forcing
    F::VV
    # Slow - Coupling strength
    G::FT
    # Fast - Coupling strength
    h::FT
    # fast timescale separation
    c::FT
    # Fast - Nonlinearity
    b::FT
end


# Forward pass of the Lorenz 96 model
#=function lorenz_forward(settings::LSettings, params::LParams)
    # run the Lorenz simulation
    xn, t = lorenz_solve(settings, params)
    # Get statistics
    gt = stats(settings, xn, t)
    return gt
end
=#

# initial run
function lorenz_solve(settings::LSettings, params::LParams)
    X = fill(Float64(0.0), 2 * settings.N, 1) #fast+slow
    if !isnothing(settings.initial)
        X = X .+ settings.initial
    end
    return lorenz_solve(X, settings, params)
end

# Solve the Lorenz 96 system 
function lorenz_solve(X::Matrix, settings::LSettings, params::LParams)
    X_next = X[:, end] #take the final state
    # Initialize
    nstep = Int32(ceil(settings.t_end / settings.dt))
    xn = zeros(Float64, 2 * settings.N, nstep)
    t = zeros(Float64, nstep)
    # March forward in time
    for j in 1:nstep
        t[j] = settings.dt * j
        X_next = RK4(X_next, settings.dt, settings.N, params, t[j])
        xn[:, j] = X_next
    end
    # Output
    return xn, t

end

# Lorenz-96 system with one substep
# f = dx/dt
# Inputs: x: state, N: longitude steps, F: forcing
function rk_inner(x, N, params, t)

    #get lorenz parameters
    (F, G, h, c, b) = (params.F, params.G, params.h, params.c, params.b)
    F_local = params.F[1] + params.F[2] * sin(params.F[3] * 2 * π * t)

    slow_id = 1:N
    fast_id = (N + 1):(2 * N)
    xs = x[slow_id]
    xf = x[fast_id]
    slow = zeros(Float64, N)
    fast = zeros(Float64, N)
    xnew = zeros(size(x))
    # Loop over N positions

    # dynamics - slow (F,G)
    #=
        # Damping system with coupling and force
        for i = 1:N
            slow[i] = - xs[i] + F_local - G*xf[i]
        end
        =#
    # Damping system with coupling and force
    for i in 3:(N - 1)
        slow[i] = -xs[i - 1] * (xs[i - 2] - xs[i + 1]) - xs[i] + F_local - G * xf[i]
    end
    # Periodic BC
    slow[2] = -xs[1] * (xs[N] - xs[3]) - xs[2] + F_local - G * xf[2]
    slow[1] = -xs[N] * (xs[N - 1] - xs[2]) - xs[1] + F_local - G * xf[1]
    slow[N] = -xs[N - 1] * (xs[N - 2] - xs[1]) - xs[N] + F_local - G * xf[N]

    # dynamics - fast (h,c,b)
    # L96 system  with  coupling to slow
    for i in 2:(N - 2)
        fast[i] = -c * b * xf[i + 1] * (xf[i + 2] - xf[i - 1]) - c * xf[i] + h * c / b * xs[i]
    end
    # periodic boundary conditions
    fast[1] = -c * b * xf[2] * (xf[3] - xf[N]) - c * xf[1] + h * c / b * xs[1]
    fast[N - 1] = -c * b * xf[N] * (xf[1] - xf[N - 2]) - c * xf[N - 1] + h * c / b * xs[N - 1]
    fast[N] = -c * b * xf[1] * (xf[2] - xf[N - 1]) - c * xf[N] + h * c / b * xs[N]

    # return
    xnew[slow_id] = slow
    xnew[fast_id] = fast
    return xnew
end

# RK4 solve
function RK4(xold, dt, N, F, t)
    # Predictor steps
    k1 = rk_inner(xold, N, F, t)
    k2 = rk_inner(xold + k1 * dt / 2.0, N, F, t)
    k3 = rk_inner(xold + k2 * dt / 2.0, N, F, t)
    k4 = rk_inner(xold + k3 * dt, N, F, t)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end
