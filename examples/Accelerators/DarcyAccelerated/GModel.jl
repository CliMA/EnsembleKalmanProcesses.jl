##################
# Copied on 3/16/23 and modified from 
# https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl
##################

using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays




mutable struct Setup_Param{FT <: AbstractFloat, IT <: Int}
    # physics
    N::IT         # number of grid points for both x and y directions (including both ends)
    Δx::FT
    xx::AbstractVector{FT}  # uniform grid [a, a+Δx, a+2Δx ... b] (in each dimension)

    #for source term
    f_2d::AbstractMatrix{FT}

    κ::AbstractMatrix{FT}

    # observation locations is tensor product x_locs × y_locs
    x_locs::AbstractVector{IT}
    y_locs::AbstractVector{IT}

    N_y::IT
end


function Setup_Param(
    xx::AbstractVector{FT},
    obs_ΔN::IT,
    κ::AbstractMatrix;
    seed::IT = 123,
) where {FT <: AbstractFloat, IT <: Int}

    N = length(xx)
    Δx = xx[2] - xx[1]

    #  logκ_2d, φ, λ, θ_ref = generate_θ_KL(xx, N_KL, d, τ, seed=seed)
    f_2d = compute_f_2d(xx)

    x_locs = Array(obs_ΔN:obs_ΔN:(N - obs_ΔN))
    y_locs = Array(obs_ΔN:obs_ΔN:(N - obs_ΔN))
    N_y = length(x_locs) * length(y_locs)

    Setup_Param(N, Δx, xx, f_2d, κ, x_locs, y_locs, N_y)
end



#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
function compute_f_2d(yy::AbstractVector{FT}) where {FT <: AbstractFloat}
    N = length(yy)
    f_2d = zeros(FT, N, N)
    for i in 1:N
        if (yy[i] <= 4 / 6)
            f_2d[:, i] .= 1000.0
        elseif (yy[i] >= 4 / 6 && yy[i] <= 5 / 6)
            f_2d[:, i] .= 2000.0
        elseif (yy[i] >= 5 / 6)
            f_2d[:, i] .= 3000.0
        end
    end
    return f_2d
end

"""
    run_G_ensemble(darcy,κs::AbstractMatrix)

Computes the forward map `G` (`solve_Darcy_2D` followed by `compute_obs`) over an ensemble of `κ`'s, stored flat as columns of `κs`
"""
function run_G_ensemble(darcy, κs::AbstractMatrix)
    N_ens = size(κs, 2) # ens size
    nd = darcy.N_y #num obs
    g_ens = zeros(nd, N_ens)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        κ_i = reshape(κs[:, i], darcy.N, darcy.N) # unflatten
        h_i = solve_Darcy_2D(darcy, κ_i) # run model
        g_ens[:, i] = compute_obs(darcy, h_i) # observe solution
    end
    return g_ens
end



#=
    return the unknow index for the grid point
    Since zero-Dirichlet boundary conditions are imposed on  
    all four edges, the freedoms are only on interior points
=#
function ind(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT <: AbstractFloat, IT <: Int}
    return (ix - 1) + (iy - 2) * (darcy.N - 2)
end

function ind_all(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT <: AbstractFloat, IT <: Int}
    return ix + (iy - 1) * darcy.N
end

#=
    solve Darcy equation with finite difference method:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
=#
function solve_Darcy_2D(darcy::Setup_Param{FT, IT}, κ_2d::AbstractMatrix{FT}) where {FT <: AbstractFloat, IT <: Int}
    Δx, N = darcy.Δx, darcy.N

    indx = IT[]
    indy = IT[]
    vals = FT[]

    f_2d = darcy.f_2d

    𝓒 = Δx^2
    for iy in 2:(N - 1)
        for ix in 2:(N - 1)

            ixy = ind(darcy, ix, iy)

            #top
            if iy == N - 1
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix, iy + 1]) / 2.0 / 𝓒)
            else
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (h_2d[ix,iy+1] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy + 1)])
                append!(
                    vals,
                    [(κ_2d[ix, iy] + κ_2d[ix, iy + 1]) / 2.0 / 𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy + 1]) / 2.0 / 𝓒],
                )
            end

            #bottom
            if iy == 2
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix, iy - 1]) / 2.0 / 𝓒)
            else
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - h_2d[ix,iy-1])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy - 1)])
                append!(
                    vals,
                    [(κ_2d[ix, iy] + κ_2d[ix, iy - 1]) / 2.0 / 𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy - 1]) / 2.0 / 𝓒],
                )
            end

            #right
            if ix == N - 1
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix + 1, iy]) / 2.0 / 𝓒)
            else
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (h_2d[ix+1,iy] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix + 1, iy)])
                append!(
                    vals,
                    [(κ_2d[ix, iy] + κ_2d[ix + 1, iy]) / 2.0 / 𝓒, -(κ_2d[ix, iy] + κ_2d[ix + 1, iy]) / 2.0 / 𝓒],
                )
            end

            #left
            if ix == 2
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix - 1, iy]) / 2.0 / 𝓒)
            else
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - h_2d[ix-1,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix - 1, iy)])
                append!(
                    vals,
                    [(κ_2d[ix, iy] + κ_2d[ix - 1, iy]) / 2.0 / 𝓒, -(κ_2d[ix, iy] + κ_2d[ix - 1, iy]) / 2.0 / 𝓒],
                )
            end

        end
    end



    df = sparse(indx, indy, vals, (N - 2)^2, (N - 2)^2)
    # Multithread does not support sparse matrix solver
    h = df \ (f_2d[2:(N - 1), 2:(N - 1)])[:]

    h_2d = zeros(FT, N, N)
    h_2d[2:(N - 1), 2:(N - 1)] .= reshape(h, N - 2, N - 2)

    return h_2d
end

#=
Compute observation values
=#
function compute_obs(darcy::Setup_Param{FT, IT}, h_2d::AbstractMatrix{FT}) where {FT <: AbstractFloat, IT <: Int}
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[darcy.x_locs, darcy.y_locs]

    return obs_2d[:]
end
