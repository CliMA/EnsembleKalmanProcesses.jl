#Ensemble Kalman Sampler: specific structures and function definitions

"""
    Sampler{FT<:AbstractFloat,IT<:Int} <: Process

An ensemble Kalman Sampler process.
"""
struct Sampler{FT <: AbstractFloat} <: Process
    ""
    prior_mean::Vector{FT}
    ""
    prior_cov::Array{FT, 2}
end



function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}}, g_in::Array{FT, 2}) where {FT, IT}

    #catch works when g_in non-square 
    if !(size(g_in)[2] == ekp.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try transposing or check ensemble size"))
    end

    # u: N_ens × N_par
    # g: N_ens × N_obs
    u_old = get_u_final(ekp)
    u_old = permutedims(u_old, (2, 1))
    u = u_old
    g = permutedims(g_in, (2, 1))

    # u_mean: N_par × 1
    u_mean = mean(u', dims = 2)
    # g_mean: N_obs × 1
    g_mean = mean(g', dims = 2)
    # g_cov: N_obs × N_obs
    g_cov = cov(g, corrected = false)
    # u_cov: N_par × N_par
    u_cov = cov(u, corrected = false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- ekp.obs_mean
    # D: N_ens × N_ens
    D = (1 / ekp.N_ens) * (E' * (ekp.obs_noise_cov \ R))

    Δt = 1 / (norm(D) + 1e-8)

    noise = MvNormal(u_cov)

    implicit =
        (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov' \ u_cov')') \
        (u' .- Δt * (u' .- u_mean) * D .+ Δt * u_cov * (ekp.process.prior_cov \ ekp.process.prior_mean))

    u = implicit' + sqrt(2 * Δt) * rand(noise, ekp.N_ens)'

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns = false))
    push!(ekp.g, DataContainer(g, data_are_columns = false))
    push!(ekp.Δt, Δt)
    # u_old is N_ens × N_par, g is N_ens × N_obs,
    # but stored in data container with N_ens as the 2nd dim

    compute_error!(ekp)

end
