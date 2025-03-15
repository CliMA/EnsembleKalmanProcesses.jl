#Unscented Transform Kalman Inversion: specific structures and function definitions

# Many functions are shared with Unscented Kalman Inversion, and are defined in `src/UnscentedKalmanInversion.jl`
"""
    TransformUnscented{FT<:AbstractFloat, IT<:Int} <: Process

An unscented Kalman Inversion process.

# Fields

$(TYPEDFIELDS)

# Constructors

    TransformUnscented(
        u0_mean::AbstractVector{FT},
        uu0_cov::AbstractMatrix{FT};
        α_reg::FT = 1.0,
        update_freq::IT = 0,
        modified_unscented_transform::Bool = true,
        impose_prior::Bool = false,
        prior_mean::Any,
        prior_cov::Any,
        sigma_points::String = symmetric
    ) where {FT <: AbstractFloat, IT <: Int}

Construct an TransformUnscented Inversion Process.

Inputs:

  - `u0_mean`: Mean at initialization.
  - `uu0_cov`: Covariance at initialization.
  - `α_reg`: Hyperparameter controlling regularization toward the prior mean (0 < `α_reg` ≤ 1),
  default should be 1, without regulariazion.
  - `update_freq`: Set to 0 when the inverse problem is not identifiable, 
  namely the inverse problem has multiple solutions, the covariance matrix
  will represent only the sensitivity of the parameters, instead of
  posterior covariance information; set to 1 (or anything > 0) when
  the inverse problem is identifiable, and the covariance matrix will
  converge to a good approximation of the posterior covariance with an
  uninformative prior.
  - `modified_unscented_transform`: Modification of the UKI quadrature given
    in Huang et al (2021).
  - `impose_prior`: using augmented system (Tikhonov regularization with Kalman inversion in Chada 
     et al 2020 and Huang et al (2022)) to regularize the inverse problem, which also imposes prior 
     for posterior estimation. If impose_prior == true, prior mean and prior cov must be provided. 
     This is recommended to use, especially when the number of observations is smaller than the number 
     of parameters (ill-posed inverse problems). When this is used, other regularizations are turned off
     automatically.
  - `prior_mean`: Prior mean used for regularization.
  - `prior_cov`: Prior cov used for regularization.
  - `sigma_points`: String of sigma point type, it can be `symmetric` with `2N_par+1` 
     ensemble members or `simplex` with `N_par+2` ensemble members.
  
$(METHODLIST)
"""
mutable struct TransformUnscented{FT <: AbstractFloat, IT <: Int} <: Process
    "an iterable of arrays of size `N_parameters` containing the mean of the parameters (in each `uki` iteration a new array of mean is added), note - this is not the same as the ensemble mean of the sigma ensemble as it is taken prior to prediction"
    u_mean::Any  # ::Iterable{AbtractVector{FT}}
    "an iterable of arrays of size (`N_parameters x N_parameters`) containing the covariance of the parameters (in each `uki` iteration a new array of `cov` is added), note - this is not the same as the ensemble cov of the sigma ensemble as it is taken prior to prediction"
    uu_cov::Any  # ::Iterable{AbstractMatrix{FT}}
    "an iterable of arrays of size `N_y` containing the predicted observation (in each `uki` iteration a new array of predicted observation is added)"
    obs_pred::Any # ::Iterable{AbstractVector{FT}}
    "weights in UKI"
    c_weights::Union{AbstractVector{FT}, AbstractMatrix{FT}}
    mean_weights::AbstractVector{FT}
    cov_weights::AbstractVector{FT}
    "number of particles 2N+1 or N+2"
    N_ens::IT
    "covariance of the artificial evolution error"
    Σ_ω::AbstractMatrix{FT}
    "covariance of the artificial observation error"
    Σ_ν_scale::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::AbstractVector{FT}
    "update frequency"
    update_freq::IT
    "using augmented system (Tikhonov regularization with Kalman inversion in Chada 
    et al 2020 and Huang et al (2022)) to regularize the inverse problem, which also imposes prior 
    for posterior estimation."
    impose_prior::Bool
    "prior mean - defaults to initial mean"
    prior_mean::Any
    "prior covariance - defaults to initial covariance"
    prior_cov::Any
    "current iteration number"
    iter::IT
    "used to store matrices: buffer[1] = Y' *Γ_inv, buffer[2] = Y' * Γ_inv * Y"
    buffer::AbstractVector
end

function TransformUnscented(
    u0_mean::VV,
    uu0_cov::MM;
    α_reg::FT = 1.0,
    update_freq::IT = 0,
    modified_unscented_transform::Bool = true,
    impose_prior::Bool = false,
    prior_mean::Any = nothing,
    prior_cov::Any = nothing,
    sigma_points::String = "symmetric",
) where {FT <: AbstractFloat, IT <: Int, VV <: AbstractVector, MM <: AbstractMatrix}

    u0_mean = FT.(u0_mean)
    uu0_cov = FT.(uu0_cov)
    if impose_prior
        if isnothing(prior_mean)
            @info "`impose_prior=true` but `prior_mean=nothing`, taking initial mean as prior mean."
            prior_mean = u0_mean
        else
            prior_mean = FT.(prior_mean)
        end
        if isnothing(prior_cov)
            @info "`impose_prior=true` but `prior_cov=nothing`, taking initial covariance as prior covariance"
            prior_cov = uu0_cov
        else
            prior_cov = FT.(prior_cov)
        end
        α_reg = 1.0
        update_freq = 1
    end

    if sigma_points == "symmetric"
        N_ens = 2 * size(u0_mean, 1) + 1
    elseif sigma_points == "simplex"
        N_ens = size(u0_mean, 1) + 2
    else
        throw(ArgumentError("sigma_points type is not recognized. Select from \"symmetric\" or \"simplex\". "))
    end

    N_par = size(u0_mean, 1)
    # ensemble size

    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    if sigma_points == "symmetric"
        c_weights = zeros(FT, N_par)

        # set parameters λ, α
        α = min(sqrt(4 / N_par), 1.0)
        λ = α^2 * N_par - N_par

        c_weights[1:N_par] .= sqrt(N_par + λ)
        mean_weights[1] = λ / (N_par + λ)
        mean_weights[2:N_ens] .= 1 / (2 * (N_par + λ))
        cov_weights[1] = λ / (N_par + λ) + 1 - α^2 + 2.0
        cov_weights[2:N_ens] .= 1 / (2 * (N_par + λ))



    elseif sigma_points == "simplex"
        c_weights = zeros(FT, N_par, N_ens)

        # set parameters λ, α
        α = N_par / (4 * (N_par + 1))

        IM = zeros(FT, N_par, N_par + 1)
        IM[1, 1], IM[1, 2] = -1 / sqrt(2α), 1 / sqrt(2α)
        for i in 2:N_par
            for j in 1:i
                IM[i, j] = 1 / sqrt(α * i * (i + 1))
            end
            IM[i, i + 1] = -i / sqrt(α * i * (i + 1))
        end
        c_weights[:, 2:end] .= IM

        mean_weights .= 1 / (N_par + 1)
        mean_weights[1] = 0.0
        cov_weights .= α
        cov_weights[1] = 0.0

    end

    if modified_unscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end

    u_mean = Vector{FT}[]  # array of Vector{FT}'s
    push!(u_mean, u0_mean) # insert parameters at end of array (in this case just 1st entry)
    uu_cov = Matrix{FT}[]  # array of Matrix{FT}'s
    push!(uu_cov, uu0_cov) # insert parameters at end of array (in this case just 1st entry)

    obs_pred = Vector{FT}[]  # array of Vector{FT}'s

    Σ_ω = (2 - α_reg^2) * uu0_cov
    Σ_ν_scale = 2.0

    r = isnothing(prior_mean) ? u0_mean : prior_mean
    iter = 0

    TransformUnscented(
        u_mean,
        uu_cov,
        obs_pred,
        c_weights,
        mean_weights,
        cov_weights,
        N_ens,
        Σ_ω,
        Σ_ν_scale,
        α_reg,
        r,
        update_freq,
        impose_prior,
        prior_mean,
        prior_cov,
        iter,
        [],
    )
end

function TransformUnscented(prior::ParameterDistribution; kwargs...)
    u0_mean = isa(mean(prior), AbstractVector) ? Vector(mean(prior)) : [mean(prior)] # mean of unconstrained distribution
    uu0_cov = Matrix(cov(prior)) # cov of unconstrained distribution

    return TransformUnscented(u0_mean, uu0_cov; prior_mean = u0_mean, prior_cov = uu0_cov, kwargs...)

end


"""
$(TYPEDSIGNATURES)

Returns the stored `buffer` from the TransformUnscented process 
"""
get_buffer(p::TU) where {TU <: TransformUnscented} = p.buffer

function FailureHandler(process::TransformUnscented, method::IgnoreFailures)
    function failsafe_update(uki, u, g, u_idx, g_idx, failed_ens)
        #perform analysis on the model runs
        update_ensemble_analysis!(uki, u, g, u_idx, g_idx)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(get_process(uki), get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{TransformUnscented, IgnoreFailures}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Provides a failsafe update that
 - computes all means and covariances over the successful sigma points,
 - rescales the mean weights and the off-center covariance weights of the
    successful particles to sum to the same value as the original weight sums.
"""
function FailureHandler(process::TransformUnscented, method::SampleSuccGauss)
    function succ_gauss_analysis!(
        uki::EnsembleKalmanProcess{FT, IT, TU},
        u_p_full,
        g_full,
        u_idx,
        g_idx,
        failed_ens,
    ) where {FT <: Real, IT <: Int, TU <: TransformUnscented}

        process = get_process(uki)
        inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν
        process = get_process(uki)

        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g_full, 2)))

        # update group index of y,g,u
        prior_mean = process.prior_mean[u_idx]
        prior_cov_inv = inv(process.prior_cov)[u_idx, u_idx] # take idx later
        u_p = u_p_full[u_idx, :]
        y = get_obs(uki)[g_idx]
        g = g_full[g_idx, :]

        u_p_mean = construct_successful_mean(uki, u_p, successful_ens)
        m = length(successful_ens)
        g_mean = construct_successful_mean(uki, g, successful_ens)

        u_p = u_p[:, successful_ens]
        g = g[:, successful_ens]

        ## extend the state (NB obs_noise_cov_inv already extended)
        if process.impose_prior
            # extend y and G
            g_ext = [g; u_p]
            g_mean_ext = [g_mean; u_p_mean]
            y_ext = [y; prior_mean]
        else
            y_ext = y
            g_mean_ext = g_mean
            g_ext = g
        end

        # sqrt-increments
        X = FT.((u_p .- u_p_mean) / sqrt(m - 1))
        Y = FT.((g_ext .- g_mean_ext) / sqrt(m - 1)) # may cause issue as first row = 0? TBD

        # Create/Enlarge buffers if needed
        tmp = get_buffer(get_process(uki)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
        ys1, ys2 = size(Y)
        if length(tmp) == 0  # no buffer
            push!(tmp, zeros(ys2, ys1)) # stores Y' * Γ_inv
            push!(tmp, zeros(ys2, ys2)) # stores Y' * Γ_inv * Y
        elseif (size(tmp[1], 1) < ys2) || (size(tmp[1], 2) < ys1) # existing buffer is too small
            tmp[1] = zeros(ys2, ys1)
            tmp[2] = zeros(ys2, ys2)
        end

        if process.impose_prior
            lmul_obs_noise_cov_inv!(view(tmp[1]', 1:size(g, 1), 1:ys2), uki, Y[1:size(g, 1), :], g_idx) # store in transpose, with view helping reduce allocations
            view(tmp[1]', (size(g, 1) + 1):ys1, 1:ys2) .= process.Σ_ν_scale * prior_cov_inv * Y[(size(g, 1) + 1):end, :]
        else
            lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, g_idx) # store in transpose, with view helping reduce allocations
        end
        view(tmp[1], 1:ys2, :) .*= inv_noise_scaling

        tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

        for i in 1:ys2
            tmp[2][i, i] += 1.0
        end
        Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y
        u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext)) #  mean update = Ω * Y' * Γ_inv * (y .- g_mean))
        uu_cov = X * Ω * X' # cov update 

        ########### Save results
        process.obs_pred[end][g_idx] .= g_mean
        process.u_mean[end][u_idx] .= u_mean
        process.uu_cov[end][u_idx, u_idx] .= uu_cov

    end
    function failsafe_update(uki, u, g, u_idx, g_idx, failed_ens)
        #perform analysis on the model runs
        succ_gauss_analysis!(uki, u, g, u_idx, g_idx, failed_ens)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(process, get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{TransformUnscented, SampleSuccGauss}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

UKI analysis step  : g is the predicted observations  `Ny x N_ens` matrix
"""
function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    u_p_full::AM1,
    g_full::AM2,
    u_idx::Vector{Int},
    g_idx::Vector{Int},
) where {
    FT <: Real,
    IT <: Int,
    TU <: TransformUnscented,
    AM1 <: AbstractMatrix,
    AM2 <: AbstractMatrix,
}

    process = get_process(uki)
    # Σ_ν = process.Σ_ν_scale * get_obs_noise_cov(uki) # inefficient
    inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν

    # update group index of y,g,u
    prior_mean = process.prior_mean[u_idx]
    prior_cov_inv = inv(process.prior_cov)[u_idx, u_idx] # take idx later
    u_p = u_p_full[u_idx, :]
    y = get_obs(uki)[g_idx]
    g = g_full[g_idx, :]

    u_p_mean = construct_mean(uki, u_p)
    m = size(u_p, 2)
    g_mean = construct_mean(uki, g)

    if process.impose_prior
        # extend y and G
        g_ext = [g; u_p]
        g_mean_ext = [g_mean; u_p_mean]
        y_ext = [y; prior_mean]
    else
        y_ext = y
        g_mean_ext = g_mean
        g_ext = g
    end

    # sqrt-increments
    X = FT.((u_p .- u_p_mean) / sqrt(m - 1))
    Y = FT.((g_ext .- g_mean_ext) / sqrt(m - 1))

    # Create/Enlarge buffers if needed
    tmp = get_buffer(get_process(uki)) # the buffer stores Y' * Γ_inv of [size(Y,2),size(Y,1)]
    ys1, ys2 = size(Y)
    if length(tmp) == 0  # no buffer
        push!(tmp, zeros(ys2, ys1)) # stores Y' * Γ_inv
        push!(tmp, zeros(ys2, ys2)) # stores Y' * Γ_inv * Y
    elseif (size(tmp[1], 1) < ys2) || (size(tmp[1], 2) < ys1) # existing buffer is too small
        tmp[1] = zeros(ys2, ys1)
        tmp[2] = zeros(ys2, ys2)
    end

    if process.impose_prior
        lmul_obs_noise_cov_inv!(view(tmp[1]', 1:size(g, 1), 1:ys2), uki, Y[1:size(g, 1), :], g_idx) # store in transpose, with view helping reduce allocations
        view(tmp[1]', (size(g, 1) + 1):ys1, 1:ys2) .=  process.Σ_ν_scale * prior_cov_inv * Y[(size(g, 1) + 1):end, :] # 1/Σ_ν_scale is in inv_noise_scaling below, so this will cancel it for this term
    else
        lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, g_idx) # store in transpose, with view helping reduce allocations
    end
    view(tmp[1], 1:ys2, :) .*= inv_noise_scaling

    tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

    for i in 1:ys2
        tmp[2][i, i] += 1.0
    end
    Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y
    u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext)) #  mean update = Ω * Y' * Γ_inv * (y .- g_mean))
    uu_cov = X * Ω * X' # cov update 

    ########### Save results
    process.obs_pred[end][g_idx] .= g_mean
    process.u_mean[end][u_idx] .= u_mean
    process.uu_cov[end][u_idx, u_idx] .= uu_cov

end

"""
$(TYPEDSIGNATURES)

Updates the ensemble according to an TransformUnscented process. 

Inputs:
 - `uki`        :: The EnsembleKalmanProcess to update.
 - `g_in`       :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - `process` :: Type of the EKP.
 - `u_idx` :: indices of u to update (see `UpdateGroup`)
 - `g_idx` :: indices of g,y,Γ with which to update u (see `UpdateGroup`)
 - `group_idx` :: the label of the update group (1 is "first update this iteration")
 - `failed_ens` :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    g_in::AbstractMatrix{FT},
    process::TU,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    group_idx = 0,
    failed_ens = nothing,
    kwargs...,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    #catch works when g_in non-square 
    u_p_old = get_u_final(uki)
    process = get_process(uki)
    
    fh = get_failure_handler(uki)

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g_in)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    # create on first group, then populate later
    if group_idx == 1
        push!(process.obs_pred, zeros(size(g_in, 1)))
        push!(process.u_mean, zeros(size(u_p_old, 1)))
        push!(process.uu_cov, zeros(size(u_p_old, 1), size(u_p_old, 1)))
    end

    u_p = fh.failsafe_update(uki, u_p_old, g_in, u_idx, g_idx, failed_ens)

    return u_p
end
