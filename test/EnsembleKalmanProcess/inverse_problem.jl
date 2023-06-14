using Distributions
using LinearAlgebra
using Random

using EnsembleKalmanProcesses.ParameterDistributions

"""
    G_linear(ϕ, A)

Linear forward map returning Aϕ
"""
function G_linear(ϕ, A)
    return A * ϕ
end

"""
    linear_map(rng, n_obs)

Randomly chosen linear map described by a MvNormal operator with nonzero correlation
"""
function linear_map(rng, n_obs)
    cross_corr = rand(rng, Uniform(-0.9, 0.9))
    C = [1 -cross_corr; -cross_corr 1]          # Correlation structure for linear operator
    return rand(rng, MvNormal(zeros(size(C, 1)), C), n_obs)'    # Linear operator in R^{n_obs x 2}
end

"""
    G_nonlinear_old(ϕ, rng, noise_dist, ϕ_star = 0.0)

Noisy nonlinear map output given `ϕ` and multivariate noise.
The map: `(||ϕ-ϕ_star^*||_1, ||ϕ-ϕ_star||_2, ||ϕ-ϕ_star||_∞)`

Inputs:

 - `ϕ`                           :: Input to the forward map (constrained space)
 - `rng`                         :: Random number generator
 - `noise_dist`                  :: Multivariate noise distribution from which stochasticity of map is sampled.
 - `ϕ_star`                      :: Minimum of the function
 - `add_or_mult_noise = nothing` :: additive ["add"] or multiplicative ["mult"] variability optional
"""
function G_nonlinear(ϕ, rng, noise_dist; ϕ_star = 0.0, add_or_mult_noise = nothing)
    if ndims(ϕ) == 1
        p = length(ϕ)
        ϕ_new = reshape(ϕ, (p, 1))
    else
        ϕ_new = ϕ
    end

    true_output = zeros(3, size(ϕ_new, 2)) #l^1, l^2, l^inf norm
    true_output[1, :] = reshape([norm(c, 1) for c in eachcol(ϕ_new .- ϕ_star)], 1, :)
    true_output[2, :] = reshape([norm(c, 2) for c in eachcol(ϕ_new .- ϕ_star)], 1, :)
    true_output[3, :] = reshape([maximum(abs.(c)) for c in eachcol(ϕ_new .- ϕ_star)], 1, :)


    # Add noise from noise_dist
    if isnothing(add_or_mult_noise)
        #just copy the observation n_obs//3 times
        output = repeat(true_output, length(noise_dist), 1) # (n_obs x n_ens)
    else
        output = zeros(length(noise_dist) * size(true_output, 1), size(true_output, 2))
        internal_noise = rand(rng, noise_dist, size(true_output, 2))

        if add_or_mult_noise == "add"
            for i in 1:length(noise_dist)
                output[(3 * (i - 1) + 1):(3 * i), :] = reshape(internal_noise[i, :], 1, :) .+ true_output
            end
        elseif add_or_mult_noise == "mult"
            for i in 1:length(noise_dist)
                output[(3 * (i - 1) + 1):(3 * i), :] = (1 .+ reshape(internal_noise[i, :], 1, :)) .* true_output
            end
        else
            throw(
                ArgumentError(
                    "`add_or_mult_noise` keyword expects either nothing, \"add\" or \"mult\". Got $(add_or_mult_noise).",
                ),
            )
        end
    end

    return size(output, 2) == 1 ? vec(output) : output
end


"""
    G_nonlinear_old(ϕ, rng, noise_dist, ϕ_star = 0.0)

Noisy nonlinear map output given `ϕ` and multivariate noise.
The map: √(sum(ϕ - ϕ_star)²)

Inputs:

 - `ϕ`                           :: Input to the forward map (constrained space)
 - `rng`                         :: Random number generator
 - `noise_dist`                  :: Multivariate noise distribution from which stochasticity of map is sampled.
 - `ϕ_star`                      :: Minimum of the function
 - `add_or_mult_noise = nothing` :: additive ["add"] or multiplicative ["mult"] variability optional

"""
function G_nonlinear_old(ϕ, rng, noise_dist; ϕ_star = 0.0, add_or_mult_noise = nothing)
    if ndims(ϕ) == 1
        p = length(ϕ)
        ϕ_new = reshape(ϕ, (p, 1))
    else
        ϕ_new = ϕ
    end
    true_output = sqrt.(sum((ϕ_new .- ϕ_star) .^ 2, dims = 1)) # (1 x n_ens)

    # Add noise from noise_dist

    if isnothing(add_or_mult_noise)
        #just copy the observation n_obs times
        output = repeat(true_output, length(noise_dist), 1) # (n_obs x n_ens)
    elseif add_or_mult_noise == "add"
        output = rand(rng, noise_dist, size(true_output, 2)) .+ true_output # (n_obs x n_ens)   
    elseif add_or_mult_noise == "mult"
        output = (1 .+ rand(rng, noise_dist)) * true_output # (n_obs x n_ens)
    else
        throw(
            ArgumentError(
                "`add_or_mult_noise` keyword expects either nothing, \"add\" or \"mult\". Got $(add_or_mult_noise).",
            ),
        )
    end
    return size(output, 2) == 1 ? vec(output) : output
end

"""
    linear_inv_problem(ϕ_star, noise_sd, n_obs, rng; obs_corrmat = I, return_matrix = false)

Returns the objects defining a linear inverse problem.

Inputs:

 - `ϕ_star`          :: True parameter value (constrained space)
 - `noise_sd`        :: Magnitude of the observational noise
 - `n_obs`           :: Dimension of the observational vector
 - `rng`             :: Random number generator
 - `obs_corrmat`     :: Correlation structure of the observational noise
 - `return_matrix`   :: If true, returns the forward map operator

Returns:

 - `y_obs`           :: Vector of observations
 - `G`               :: Map from constrained space to data space
 - `Γ`               :: Observational covariance matrix
 - `A`               :: Forward model operator from constrained space
"""
function linear_inv_problem(ϕ_star, noise_sd, n_obs, rng; obs_corrmat = I, return_matrix = false)
    A = linear_map(rng, n_obs)
    # Define forward map from unconstrained space directly
    G(x) = G_linear(x, A)
    y_star = G(ϕ_star)
    Γ = noise_sd^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    y_obs = y_star .+ rand(rng, noise)
    if return_matrix
        return y_obs, G, Γ, A
    else
        return y_obs, G, Γ
    end
end

"""
    nonlinear_inv_problem(ϕ_star, noise_sd, n_obs, rng; obs_corrmat = I; add_or_mult_noise = "add")

Returns the objects defining a random nonlinear inverse problem 

Inputs:

 - `ϕ_star`                      :: True parameter value (constrained space)
 - `noise_sd`                    :: Magnitude of the observational noise
 - `n_obs`                       :: Dimension of the observational vector
 - `rng`                         :: Random number generator
 - `obs_corrmat = I  `           :: Correlation structure of the observational noise 
 - `add_or_mult_noise = nothing` :: additive ["add"] or multiplicative ["mult"] variability optional
Returns:

 - `y_obs`           :: Vector of observations
 - `G`               :: Map from constrained space to data space
 - `Γ`               :: Observational covariance matrix
"""
function nonlinear_inv_problem(ϕ_star, noise_sd, n_obs, rng; obs_corrmat = I, add_or_mult_noise = nothing)
    Γ = noise_sd^2 * obs_corrmat
    # G produces a 3D output so must div by 3
    @assert n_obs % 3 == 0

    noise = MvNormal(zeros(n_obs), Γ)
    internal_noise = MvNormal(zeros(Int(n_obs // 3)), noise_sd^2 * I)
    # Define forward map from unconstrained space directly
    G(x) = G_nonlinear(x, rng, internal_noise, ϕ_star = ϕ_star, add_or_mult_noise = add_or_mult_noise)
    # Get observation
    y_star = G(ϕ_star)
    y_obs = y_star .+ rand(rng, noise)
    return y_obs, G, Γ
end

function nonlinear_inv_problem_old(ϕ_star, noise_sd, n_obs, rng; obs_corrmat = I, add_or_mult_noise = nothing)
    Γ = noise_sd^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    # Define forward map from unconstrained space directly
    G(x) = G_nonlinear_old(x, rng, noise, ϕ_star = ϕ_star, add_or_mult_noise = add_or_mult_noise)
    # Get observation
    y_star = G(ϕ_star)
    y_obs = y_star .+ rand(rng, noise)
    return y_obs, G, Γ
end


function plot_inv_problem_ensemble(prior, ekp, filepath)
    gr()
    ϕ_prior = transform_unconstrained_to_constrained(prior, get_u_prior(ekp))
    ϕ_final = get_ϕ_final(prior, ekp)
    p = plot(ϕ_prior[1, :], ϕ_prior[2, :], seriestype = :scatter, label = "Initial ensemble")
    plot!(ϕ_final[1, :], ϕ_final[2, :], seriestype = :scatter, label = "Final ensemble")

    plot!(
        [ϕ_star[1]],
        xaxis = "cons_p",
        yaxis = "uncons_p",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = :none,
    )
    plot!([ϕ_star[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = :none)
    savefig(p, filepath)
end
