#Unscented Kalman Inversion: specific structures and function definitions

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
    "an interable of arrays of size `N_parameters` containing the mean of the parameters (in each `uki` iteration a new array of mean is added), note - this is not the same as the ensemble mean of the sigma ensemble as it is taken prior to prediction"
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

    u0_mean = Vector(mean(prior)) # mean of unconstrained distribution
    uu0_cov = Matrix(cov(prior)) # cov of unconstrained distribution
    
    return TransformUnscented(u0_mean, uu0_cov; prior_mean = u0_mean, prior_cov = uu0_cov, kwargs...)

end


"""
$(TYPEDSIGNATURES)

Returns the stored `buffer` from the TransformUnscented process 
"""
get_buffer(p::TU) where {TU <: TransformUnscented} = p.buffer


# Special constructor for UKI Object
function EnsembleKalmanProcess(
    observation_series::OS,
    process::TransformUnscented{FT, IT};
    kwargs...,
) where {FT <: AbstractFloat, IT <: Int, OS <: ObservationSeries}
    # use the distribution stored in process to generate initial ensemble
    init_params = update_ensemble_prediction!(process, 0.0)
    return EnsembleKalmanProcess(init_params, observation_series, process; kwargs...)
end

function EnsembleKalmanProcess(
    observation::OB,
    process::TransformUnscented{FT, IT};
    kwargs...,
) where {FT <: AbstractFloat, IT <: Int, OB <: Observation}

    observation_series = ObservationSeries(observation)
    return EnsembleKalmanProcess(observation_series, process; kwargs...)
end


function EnsembleKalmanProcess(
    obs_mean::AbstractVector{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    process::TransformUnscented{FT, IT};
    kwargs...,
) where {FT <: AbstractFloat, IT <: Int}

    observation = Observation(Dict("samples" => obs_mean, "covariances" => obs_noise_cov, "names" => "observation"))
    return EnsembleKalmanProcess(observation, process; kwargs...)
end

function FailureHandler(process::TransformUnscented, method::IgnoreFailures)
    function failsafe_update(uki, u, g,  u_idx, g_idx, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean)
        #perform analysis on the model runs
        update_ensemble_analysis!(uki, u, g, u_idx, g_idx, obs_noise_cov_inv, onci_idx, prior_mean)
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
    function succ_gauss_analysis!(uki::EnsembleKalmanProcess{FT,IT,TU}, u_p, g, u_idx, g_idx, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean) where {FT <: Real, IT <: Int, TU <: TransformUnscented}

        y = get_obs(uki)
        process = get_process(uki)
        inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν
        process = get_process(uki)

        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        
        ############# Prediction step
        u_p_mean = construct_successful_mean(uki, u_p, successful_ens)
        uu_p_cov = construct_successful_cov(uki, u_p, u_p_mean, successful_ens)
        m = size(u_p, 2)
        
        ###########  Analysis step
        g_mean = construct_successful_mean(uki, g, successful_ens)


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

        #=
        # TODO: Weirdly the covariance in the below is not the prior cov but the sum of (cov_u + prior_cov)!
        if process.impose_prior
            ug_cov_reg = [ug_cov uu_p_cov]
            gg_cov_reg = [gg_cov ug_cov'; ug_cov uu_p_cov+process.prior_cov / get_Δt(uki)[end]]
            tmp = ug_cov_reg / gg_cov_reg
            u_mean = u_p_mean + tmp * [obs_mean - g_mean; process.prior_mean - u_p_mean]
            uu_cov = uu_p_cov - tmp * ug_cov_reg'
        else
            tmp = ug_cov / gg_cov
            u_mean = u_p_mean + tmp * (obs_mean - g_mean)
            uu_cov = uu_p_cov - tmp * ug_cov'
        end
        =#

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
        
        lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, onci_idx) # store in transpose, with view helping reduce allocations
        view(tmp[1], 1:ys2, :) .*= inv_noise_scaling
        
        tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y
        
        for i in 1:ys2
            tmp[2][i, i] += 1.0
        end
        Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y
        @info Ω
        u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext)) #  mean update = Ω * Y' * Γ_inv * (y .- g_mean))
        uu_cov = X*Ω*X' # cov update 

        
        ########### Save results
        push!(process.obs_pred, g_mean) # N_ens x N_data
        push!(process.u_mean, u_mean) # N_ens x N_params
        push!(process.uu_cov, uu_cov) # N_ens x N_data

    end
    function failsafe_update(uki, u, g, u_idx, g_idx, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean)
        #perform analysis on the model runs
        succ_gauss_analysis!(uki, u, g, u_idx, g_idx, obs_noise_cov_inv, onci_idx, failed_ens, prior_mean)
        #perform new prediction output to model parameters u_p
        u_p = update_ensemble_prediction!(process, get_Δt(uki)[end], u_idx)
        return u_p
    end
    return FailureHandler{TransformUnscented, SampleSuccGauss}(failsafe_update)
end

"""
$(TYPEDSIGNATURES)

Construct the sigma ensemble based on the mean `x_mean` and covariance `x_cov`.
"""
function construct_sigma_ensemble(
    process::TransformUnscented,
    x_mean::AbstractVector{FT},
    x_cov::AbstractMatrix{FT},
) where {FT <: AbstractFloat}

    N_x = size(x_mean, 1)
    N_ens = process.N_ens
    c_weights = process.c_weights

    # compute cholesky factor L of x_cov
    local chol_xx_cov
    try
        chol_xx_cov = cholesky(Hermitian(x_cov)).L
    catch
        _, S, Ut = svd(x_cov)
        # find the first singular value that is smaller than 1e-8
        ind_0 = searchsortedfirst(S, 1e-8, rev = true)
        S[ind_0:end] .= S[ind_0 - 1]
        chol_xx_cov = (qr(sqrt.(S) .* Ut).R)'
    end

    x = zeros(FT, N_x, N_ens)
    x[:, 1] = x_mean

    if isa(c_weights, AbstractVector{FT})
        for i in 1:N_x
            x[:, i + 1] = x_mean + c_weights[i] * chol_xx_cov[:, i]
            x[:, i + 1 + N_x] = x_mean - c_weights[i] * chol_xx_cov[:, i]
        end
    elseif isa(c_weights, AbstractMatrix{FT})
        for i in 2:(N_x + 2)
            x[:, i] = x_mean + chol_xx_cov * c_weights[:, i]
        end
    end

    return x
end


"""
$(TYPEDSIGNATURES)

constructs mean `x_mean` from an ensemble `x`.
"""
function construct_mean(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractVecOrMat{FT};
    mean_weights = get_process(uki).mean_weights,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}

    if isa(x, AbstractMatrix{FT})
        @assert size(x, 2) == length(mean_weights)
        return Array((mean_weights' * x')')
    else
        @assert length(mean_weights) == length(x)
        return mean_weights' * x
    end
end

"""
$(TYPEDSIGNATURES)

Constructs mean over successful particles by rescaling the quadrature
weights over the successful particles. If the central particle fails
in a modified unscented transform, the mean is computed as the
ensemble mean over all successful particles.
"""
function construct_successful_mean(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractVecOrMat{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}

    mean_weights = deepcopy(get_process(uki).mean_weights)
    # Check if modified
    if sum(mean_weights[2:end]) ≈ 0 && !(1 in successful_indices)
        mean_weights .= 1 / length(successful_indices)
    else
        mean_weights = mean_weights ./ sum(mean_weights[successful_indices])
    end
    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_mean(uki, x_succ; mean_weights = mean_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

Constructs covariance `xx_cov` from ensemble `x` and mean `x_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing} = nothing;
    cov_weights = get_process(uki).cov_weights,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}

    x_mean = isnothing(x_mean) ? construct_mean(uki, x) : x_mean

    if isa(x, AbstractMatrix{FT})
        @assert isa(x_mean, AbstractVector{FT})
        N_x, N_ens = size(x)
        xx_cov = zeros(FT, N_x, N_x)

        for i in 1:N_ens
            xx_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (x[:, i] - x_mean)'
        end
    else
        @assert isa(x_mean, FT)
        N_ens = length(x)
        xx_cov = FT(0)

        for i in 1:N_ens
            xx_cov += cov_weights[i] * (x[i] - x_mean) * (x[i] - x_mean)
        end
    end
    return xx_cov
end

"""
$(TYPEDSIGNATURES)

Constructs variance of `x` over successful particles by rescaling the
off-center weights over the successful off-center particles.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractVecOrMat{FT},
    x_mean::Union{FT, AbstractVector{FT}, Nothing},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}

    cov_weights = deepcopy(get_process(uki).cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    return construct_cov(uki, x_succ, x_mean; cov_weights = cov_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

Constructs covariance `xy_cov` from ensemble x and mean `x_mean`, ensemble `obs_mean` and mean `y_mean`.
"""
function construct_cov(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT};
    cov_weights = get_process(uki).cov_weights,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}

    N_x, N_ens = size(x)
    N_y = length(y_mean)
    xy_cov = zeros(FT, N_x, N_y)

    for i in 1:N_ens
        xy_cov .+= cov_weights[i] * (x[:, i] - x_mean) * (obs_mean[:, i] - y_mean)'
    end

    return xy_cov
end

"""
$(TYPEDSIGNATURES)

Constructs covariance of `x` and `obs_mean - y_mean` over successful particles by rescaling
the off-center weights over the successful off-center particles.
"""
function construct_successful_cov(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    x::AbstractMatrix{FT},
    x_mean::AbstractVector{FT},
    obs_mean::AbstractMatrix{FT},
    y_mean::AbstractVector{FT},
    successful_indices::Union{AbstractVector{IT}, AbstractVector{Any}},
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    N_ens, N_x, N_y = get_N_ens(uki), length(x_mean), length(y_mean)

    cov_weights = deepcopy(get_process(uki).cov_weights)

    # Rescale non-center sigma weights to sum to original value
    orig_weight_sum = sum(cov_weights[2:end])
    sum_indices = filter(x -> x > 1, successful_indices)
    succ_weight_sum = sum(cov_weights[sum_indices])
    cov_weights[2:end] = cov_weights[2:end] .* (orig_weight_sum / succ_weight_sum)

    x_succ = isa(x, AbstractMatrix) ? x[:, successful_indices] : x[successful_indices]
    obs_mean_succ = isa(x, AbstractMatrix) ? obs_mean[:, successful_indices] : obs_mean[successful_indices]
    return construct_cov(uki, x_succ, x_mean, obs_mean_succ, y_mean; cov_weights = cov_weights[successful_indices])
end

"""
$(TYPEDSIGNATURES)

UKI prediction step : generate sigma points.
"""
function update_ensemble_prediction!(process::TransformUnscented, Δt::FT, u_idx::Vector{Int}) where {FT <: AbstractFloat}

    process.iter += 1
    # update evolution covariance matrix
    if process.update_freq > 0 && process.iter % process.update_freq == 0
        process.Σ_ω[u_idx,u_idx] = (2 - process.α_reg^2) * process.uu_cov[end][u_idx,u_idx]
    end

    u_mean = process.u_mean[end][u_idx]
    uu_cov = process.uu_cov[end][u_idx,u_idx]

    α_reg = process.α_reg
    r = process.r[u_idx]
    Σ_ω = process.Σ_ω[u_idx,u_idx]

    N_par = length(u_mean[1])
    ############# Prediction step:

    u_p_mean = α_reg * u_mean + (1 - α_reg) * r
    uu_p_cov = α_reg^2 * uu_cov + Σ_ω * Δt

    ############ Generate sigma points
    u_p = construct_sigma_ensemble(process, u_p_mean, uu_p_cov)
    return u_p
end

update_ensemble_prediction!(process::TransformUnscented, Δt::FT) where {FT <: AbstractFloat} = update_ensemble_prediction!(process, Δt, collect(1:length(process.u_mean[end])))

"""
$(TYPEDSIGNATURES)

UKI analysis step  : g is the predicted observations  `Ny x N_ens` matrix
"""
function update_ensemble_analysis!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    u_p::AM1,
    g::AM2,
    u_idx::Vector{Int},
    g_idx::Vector{Int},
    obs_noise_cov_inv::AV,    
    onci_idx::AV2,
    prior_mean::NorAV,
) where {FT <: Real, IT <: Int, TU <: TransformUnscented, AM1 <: AbstractMatrix, AM2 <: AbstractMatrix, AV <: AbstractVector, AV2 <: AbstractVector, NorAV <: Union{Nothing,AbstractVector}}
    

    y = get_obs(uki)
    process = get_process(uki)
   # Σ_ν = process.Σ_ν_scale * get_obs_noise_cov(uki) # inefficient
    inv_noise_scaling = get_Δt(uki)[end] / process.Σ_ν_scale #   multiplies the inverse Σ_ν
    
    ############# Prediction step: [rebuilding where g was evaluated]

    u_p_mean = construct_mean(uki, u_p)
    uu_p_cov = construct_cov(uki, u_p, u_p_mean)
    m = size(u_p, 2)

    ###########  Analysis step

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
    
    lmul_obs_noise_cov_inv!(view(tmp[1]', :, 1:ys2), uki, Y, onci_idx) # store in transpose, with view helping reduce allocations
    view(tmp[1], 1:ys2, :) .*= inv_noise_scaling
    
    tmp[2][1:ys2, 1:ys2] = tmp[1][1:ys2, 1:ys1] * Y

    for i in 1:ys2
        tmp[2][i, i] += 1.0
    end
    Ω = inv(tmp[2][1:ys2, 1:ys2]) # Ω = (I + Y' * Γ_inv * Y)^-1 = I - Y' (Y Y' + Γ_inv)^-1 Y
    u_mean = u_p_mean + X * FT.(Ω * tmp[1][1:ys2, 1:ys1] * (y_ext .- g_mean_ext)) #  mean update = Ω * Y' * Γ_inv * (y .- g_mean))
    uu_cov = (m - 1) * X*Ω*X' # cov update 
    
    ########### Save results
    push!(process.obs_pred, g_mean) # N_ens x N_data
    push!(process.u_mean, u_mean) # N_ens x N_params
    push!(process.uu_cov, uu_cov) # N_ens x N_data

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
 - `failed_ens` :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    g_in::AbstractMatrix{FT},
    process::TU,
    u_idx::Vector{Int},
    g_idx::Vector{Int};
    failed_ens = nothing,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    #catch works when g_in non-square 
    u_p_old = get_u_final(uki)
    obs_noise_cov_inv = get_obs_noise_cov_inv(uki, build = false)# NEVER build=true for this - ruins scaling.
    process = get_process(uki)
    impose_prior = process.impose_prior
    if impose_prior # these quantitites are truncated to with onci_idx
        prior_mean = process.prior_mean
        prior_cov_inv = inv(process.prior_cov)
    else
        prior_mean = nothing
    end

    # need to sweep over local blocks
    if impose_prior
        # extend noise_cov_inv must make copy due to typing
        obs_noise_cov_inv_tmp = Vector{AbstractMatrix}(obs_noise_cov_inv)
        push!(obs_noise_cov_inv_tmp, prior_cov_inv) # extend noise cov inv to include prior cov inv
    else
        obs_noise_cov_inv_tmp = obs_noise_cov_inv
    end

    γ_sizes = zeros(Int, length(obs_noise_cov_inv))
    for (i, γ_inv) in enumerate(obs_noise_cov_inv)
        if isa(γ_inv, SVD)
            if size(γ_inv.U, 1) == size(γ_inv.U, 2)
                γ_sizes[i] = size(γ_inv.Vt, 2)
            else
                γ_sizes[i] = size(γ_inv.U, 1)
            end
        else
            γ_sizes[i] = size(γ_inv, 1)
        end
    end
    prior_flag = repeat([false], length(γ_sizes))
    if impose_prior
        prior_flag[end] = true # needed to swap g_idx to u_idx in loop
    end

    onci_idx = []
    shift = 0
    for (block_id, (γs, pf)) in enumerate(zip(γ_sizes, prior_flag))
        loc_idx = !(pf) ? intersect(1:γs, g_idx .- shift) : intersect(1:γs, u_idx)
        if !(length(loc_idx) == 0)
            push!(onci_idx, (block_id, loc_idx, loc_idx .+ shift))
        end
        shift += γs
    end
    
    fh = get_failure_handler(uki)

    if isnothing(failed_ens)
        _, failed_ens = split_indices_by_success(g_in)
    end
    if !isempty(failed_ens)
        @info "$(length(failed_ens)) particle failure(s) detected. Handler used: $(nameof(typeof(fh).parameters[2]))."
    end

    u_p = fh.failsafe_update(uki, u_p_old, g_in, u_idx, g_idx, obs_noise_cov_inv_tmp, onci_idx, failed_ens, prior_mean)

    return u_p
end

"""
$(TYPEDSIGNATURES)

Returns the mean unconstrained parameter at the requested iteration.
"""
function get_u_mean(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    iteration::IT,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    return get_process(uki).u_mean[iteration]
end

"""
$(TYPEDSIGNATURES)

Returns the unconstrained parameter covariance at the requested iteration.
"""
function get_u_cov(
    uki::EnsembleKalmanProcess{FT, IT, TU},
    iteration::IT,
) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    return get_process(uki).uu_cov[iteration]
end

function compute_error!(uki::EnsembleKalmanProcess{FT, IT, TU}) where {FT <: AbstractFloat, IT <: Int, TU <: TransformUnscented}
    mean_g = get_process(uki).obs_pred[end]
    diff = get_obs(uki) - mean_g
    X = lmul_obs_noise_cov_inv(uki, diff)
    newerr = dot(diff, X)
    push!(get_error(uki), newerr)
end


